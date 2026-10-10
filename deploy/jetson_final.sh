#!/bin/bash
# =====================================================================
# 最终部署方案：在 Jetson 上一次性完成三件事
#   1) 用 3 个新 ONNX 构建 FP16 引擎（旧引擎移到 old_engines/ 保留，不删除）：
#        fast          ← esnet_fast_v2_acfix.onnx          新 Fast（1/8 一级修正 + 双线性）
#        balanced      ← esnet_balanced_pruned_acfix.onnx  裁剪后的 Balanced
#        balancedfull  ← esnet_balanced_plugin_norange_trtfix_acfix.onnx  未裁剪的 Balanced（对比用）
#   2) 测时延 + 逐层耗时 + 内存 + 功耗（一次完成，不用再单独测功耗）
#   3) 一致性检查：新、旧引擎在 consistency_inputs/ 的真实图像上各跑一遍，导出视差
#
# 用法：把本脚本、上面 3 个 ONNX、新的 consistency_inputs/ 文件夹放到 ~/Desktop/test_floder/
#   cd ~/Desktop/test_floder && bash jetson_final.sh
#   开头输一次 sudo 密码；全程约 1–1.5 小时（建引擎最慢），期间不要运行别的程序
#   如果当时还没有 consistency_inputs/，会先跳过一致性检查；之后拷过来再运行
#     bash jetson_final.sh consistency
#   就只补做一致性检查（不会重建引擎、不会重测时延）
# 结束后：
#   - 把 final_results.tar.gz 发给我（只有日志，体积小）
#   - 把 consistency_outputs/ 文件夹拷回电脑，运行 compare_consistency.py
# =====================================================================
set -u
cd "$(dirname "$0")"
WORK=$(pwd)
TRTEXEC=/usr/src/tensorrt/bin/trtexec
PLUGIN="$WORK/libmmdeploy_tensorrt_ops.so"
OUT="$WORK/final_results"
COUT="$WORK/consistency_outputs"
mkdir -p "$OUT" "$COUT" "$WORK/old_engines"
MODES="fast balanced balancedfull"
ONLY_CONSISTENCY=0
[ "${1:-}" = "consistency" ] && ONLY_CONSISTENCY=1
declare -A ONNX=( [fast]=esnet_fast_v2_acfix.onnx
                  [balanced]=esnet_balanced_pruned_acfix.onnx
                  [balancedfull]=esnet_balanced_plugin_norange_trtfix_acfix.onnx )
declare -A WS=( [fast]=1024 [balanced]=1024 [balancedfull]=1536 )

for m in $MODES; do
  [ -f "${ONNX[$m]}" ] || { echo "找不到 ${ONNX[$m]}，请先把它拷到 $WORK"; exit 1; }
done

sudo -v || exit 1
( while true; do sudo -n true; sleep 50; done ) 2>/dev/null &   # 保持 sudo 不过期
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null; sudo pkill tegrastats 2>/dev/null' EXIT

pack() { tar czf "$WORK/final_results.tar.gz" -C "$WORK" final_results; }

echo "===== 0/4 平台与内存 ====="
{ cat /etc/nv_tegra_release; sudo nvpmodel -q; sudo jetson_clocks --show; free -m; swapon --show; } \
    > "$OUT/platform_info.txt" 2>&1
SWAP_TOTAL=$(free -m | awk '/Swap/{print $2}')
if [ "$SWAP_TOTAL" -lt 3000 ]; then   # 建大引擎时内存紧张：如果以前建过交换文件，就临时启用
  for f in /swapfile /mnt/swapfile /mnt/4GB.swap /var/swapfile; do
    if [ -f "$f" ]; then sudo swapon "$f" 2>/dev/null && echo "已临时启用交换文件 $f"; break; fi
  done
fi
free -m | grep -E "Mem|Swap"

echo "===== 1/4 构建引擎（每个约 10–30 分钟）====="
for old in esnet_fast_fp16.engine esnet_balanced_fp16.engine esnet_accurate_fp16.engine; do
  if [ -f "$old" ] && [ ! -f "old_engines/$old" ]; then
    mv -n "$old" "old_engines/$old" && echo "旧引擎已移到 old_engines/$old"
  fi
done
[ $ONLY_CONSISTENCY = 1 ] && echo "只做一致性检查，跳过第 1、2 步"
for m in $MODES; do
  [ $ONLY_CONSISTENCY = 1 ] && break
  E="esnet_${m}_fp16.engine"
  if [ -f "$E" ] && [ ! -f "old_engines/$E" ]; then
    mv -n "$E" "old_engines/$E" && echo "旧引擎已移到 old_engines/$E"
  fi
  if [ -f "$E" ] && grep -q "PASSED" "$OUT/build_${m}.log" 2>/dev/null; then
    echo "$m 已经建好，跳过"; continue
  fi
  echo "----- 建 $m ← ${ONNX[$m]}   开始 $(date +%H:%M)"
  "$TRTEXEC" --onnx="${ONNX[$m]}" --saveEngine="$E" --fp16 --workspace=${WS[$m]} \
      --plugins="$PLUGIN" > "$OUT/build_${m}.log" 2>&1 < /dev/null
  if ! grep -q "PASSED" "$OUT/build_${m}.log"; then
    echo "  第一次失败，改用 --tacticSources=-CUBLAS_LT 重试"
    mv "$OUT/build_${m}.log" "$OUT/build_${m}_try1.log"
    "$TRTEXEC" --onnx="${ONNX[$m]}" --saveEngine="$E" --fp16 --workspace=${WS[$m]} \
        --tacticSources=-CUBLAS_LT --plugins="$PLUGIN" > "$OUT/build_${m}.log" 2>&1 < /dev/null
  fi
  if ! grep -q "PASSED" "$OUT/build_${m}.log"; then
    echo "!! $m 建引擎失败。请把 final_results.tar.gz 发给我"; pack; exit 1
  fi
  echo "  完成 $(date +%H:%M)"
done

echo "===== 2/4 测时延、逐层耗时、内存、功耗 ====="
for m in $MODES; do
  [ $ONLY_CONSISTENCY = 1 ] && break
  sleep 20   # 降温
  echo "----- $m"
  sudo tegrastats --interval 100 > "$OUT/${m}_tegrastats.log" &
  sleep 3
  "$TRTEXEC" --loadEngine="esnet_${m}_fp16.engine" --plugins="$PLUGIN" \
      --warmUp=500 --iterations=100 --avgRuns=100 \
      --dumpProfile --separateProfileRun \
      --exportProfile="$OUT/${m}_profile.json" --exportTimes="$OUT/${m}_times.json" \
      > "$OUT/${m}_trtexec.log" 2>&1 < /dev/null
  sleep 3
  sudo pkill tegrastats
  grep -E "GPU Compute Time" "$OUT/${m}_trtexec.log"
done
pack

echo "===== 3/4 一致性检查（新、旧引擎各跑一遍真实图像）====="
if [ -f consistency_inputs/names.txt ]; then
  for m in $MODES; do
    [ -f "esnet_${m}_fp16.engine" ] || { echo "找不到 esnet_${m}_fp16.engine，请先不带参数运行一次本脚本"; continue; }
    for tag in new old; do
      if [ "$tag" = new ]; then E="esnet_${m}_fp16.engine"; else E="old_engines/esnet_${m}_fp16.engine"; fi
      [ "$tag" = old ] && [ "$m" != balanced ] && continue   # 旧引擎只对比 balanced（看采样属性修正前后的差异）
      [ -f "$E" ] || continue
      D="$COUT/${tag}_${m}"; mkdir -p "$D"
      echo "----- ${tag}_${m}   $(date +%H:%M)"
      while read -r stem; do
        stem=${stem%$'\r'}; [ -z "$stem" ] && continue
        "$TRTEXEC" --loadEngine="$E" --plugins="$PLUGIN" \
            --loadInputs="left:consistency_inputs/left_${stem}.bin,right:consistency_inputs/right_${stem}.bin" \
            --iterations=1 --warmUp=0 --duration=0 \
            --exportOutput="$D/${stem}.json" > "$D/${stem}.log" 2>&1 < /dev/null
        [ -s "$D/${stem}.json" ] && echo -n "." || echo -n "x"
      done < consistency_inputs/names.txt
      echo
    done
  done
else
  echo "没有找到 consistency_inputs/names.txt，先跳过一致性检查。"
  echo "等电脑上生成 consistency_inputs/ 并拷过来后，运行：bash jetson_final.sh consistency"
fi

echo "===== 4/4 打包 ====="
ls "$COUT"/*/ 2>/dev/null | grep -c json > "$OUT/consistency_json_count.txt"
pack
echo "完成！"
echo "  1) 把 $WORK/final_results.tar.gz 发给我"
echo "  2) 把 $COUT 整个文件夹拷回电脑，和 consistency_inputs 放在一起，运行 compare_consistency.py"
