#!/bin/bash
# =====================================================================
# Model A 的 Fast 引擎：构建 + 测时延/逐层耗时/内存/功耗 + 一致性检查（约 20–30 分钟）
# 必须在 jetson_final.sh 跑完之后再运行（同时跑会互相影响测速）。
#
# 用法：把 fastA_to_jetson.tar.gz、本脚本、compare_consistency.py 放到 ~/Desktop/test_floder/
#   cd ~/Desktop/test_floder
#   tar xzf fastA_to_jetson.tar.gz        # 解出 esnet_fastA_acfix.onnx 和 consistency_inputs/pt_fastA_*.npy
#   sed -i 's/\r$//' jetson_fastA.sh
#   bash jetson_fastA.sh
# 结束后把 fastA_results.tar.gz 发给我（里面已含全部引擎的一致性对比表）。
# =====================================================================
set -u
cd "$(dirname "$0")"
WORK=$(pwd)
TRTEXEC=/usr/src/tensorrt/bin/trtexec
PLUGIN="$WORK/libmmdeploy_tensorrt_ops.so"
OUT="$WORK/final_results"
COUT="$WORK/consistency_outputs"
mkdir -p "$OUT" "$COUT"
m=fastA
ONNX=esnet_fastA_acfix.onnx
E=esnet_fastA_fp16.engine

[ -f "$ONNX" ] || { echo "找不到 $ONNX，请先解压 fastA_to_jetson.tar.gz"; exit 1; }
[ -f "$PLUGIN" ] || { echo "找不到插件 $PLUGIN"; exit 1; }
ls consistency_inputs/pt_fastA_*.npy > /dev/null 2>&1 || { echo "consistency_inputs/ 里没有 pt_fastA_*.npy，请先解压 fastA_to_jetson.tar.gz"; exit 1; }
if pgrep -f jetson_final.sh > /dev/null; then echo "jetson_final.sh 还在运行，请等它结束后再运行本脚本"; exit 1; fi

sudo -v || exit 1
( while true; do sudo -n true; sleep 50; done ) 2>/dev/null &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null; sudo pkill tegrastats 2>/dev/null' EXIT
pack() { tar czf "$WORK/fastA_results.tar.gz" -C "$WORK" final_results; }

echo "===== 1/3 构建引擎（约 10–20 分钟） 开始 $(date +%H:%M) ====="
if [ -f "$E" ] && grep -q "PASSED" "$OUT/build_${m}.log" 2>/dev/null; then
  echo "已经建好，跳过"
else
  "$TRTEXEC" --onnx="$ONNX" --saveEngine="$E" --fp16 --workspace=1024 \
      --plugins="$PLUGIN" > "$OUT/build_${m}.log" 2>&1 < /dev/null
  if ! grep -q "PASSED" "$OUT/build_${m}.log"; then
    echo "  第一次失败，改用 --tacticSources=-CUBLAS_LT 重试"
    mv "$OUT/build_${m}.log" "$OUT/build_${m}_try1.log"
    "$TRTEXEC" --onnx="$ONNX" --saveEngine="$E" --fp16 --workspace=1024 \
        --tacticSources=-CUBLAS_LT --plugins="$PLUGIN" > "$OUT/build_${m}.log" 2>&1 < /dev/null
  fi
  if ! grep -q "PASSED" "$OUT/build_${m}.log"; then
    echo "!! 建引擎失败，请把 fastA_results.tar.gz 发给我"; pack; exit 1
  fi
  echo "  完成 $(date +%H:%M)"
fi

echo "===== 2/3 测时延、逐层耗时、内存、功耗 ====="
sleep 20
sudo tegrastats --interval 100 > "$OUT/${m}_tegrastats.log" &
sleep 3
"$TRTEXEC" --loadEngine="$E" --plugins="$PLUGIN" \
    --warmUp=500 --iterations=100 --avgRuns=100 \
    --dumpProfile --separateProfileRun \
    --exportProfile="$OUT/${m}_profile.json" --exportTimes="$OUT/${m}_times.json" \
    > "$OUT/${m}_trtexec.log" 2>&1 < /dev/null
sleep 3
sudo pkill tegrastats
grep -E "GPU Compute Time" "$OUT/${m}_trtexec.log"

echo "===== 3/3 一致性检查 ====="
D="$COUT/new_${m}"; mkdir -p "$D"
while read -r stem; do
  stem=${stem%$'\r'}; [ -z "$stem" ] && continue
  "$TRTEXEC" --loadEngine="$E" --plugins="$PLUGIN" \
      --loadInputs="left:consistency_inputs/left_${stem}.bin,right:consistency_inputs/right_${stem}.bin" \
      --iterations=1 --warmUp=0 --duration=0 \
      --exportOutput="$D/${stem}.json" > "$D/${stem}.log" 2>&1 < /dev/null
  [ -s "$D/${stem}.json" ] && echo -n "." || echo -n "x"
done < consistency_inputs/names.txt
echo
if [ -f compare_consistency.py ]; then
  python3 compare_consistency.py | tee "$OUT/consistency_table.txt"
else
  echo "没有 compare_consistency.py，跳过对比表（之后补做：python3 compare_consistency.py）"
fi
pack
echo "完成！把 $WORK/fastA_results.tar.gz 发给我"
