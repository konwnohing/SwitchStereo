"""
为 Model A 生成 Jetson 用的 Fast 部署模型（不用重新导出，CPU 即可，无卡模式开机也能跑，约 3–5 分钟）。

做四件事：
  1) 改图：从已修正采样属性的 Balanced ONNX（esnet_balanced_plugin_norange_trtfix_acfix.onnx）出发，
     只保留“匹配模块输出的 1/8 视差 → 双线性放大 ×8”，与 Model A 的前向计算完全一致；无用节点全部删除。
     原来的 norange / trtfix / 插件属性修正全部保留。
  2) 换权重：图里的权重来自最终模型（ohem+spatial）。先找出每个权重在最终模型里的来源
     （同名、同值，或 ResNet 中 Conv 与 BN 融合后的值），再用 Model A 的对应权重按同样方式算出新值替换。
  3) 校验：
     - 用 PyTorch 跑 Model A，得到 10 张一致性检查图像上的参考输出，保存为 consistency_inputs/pt_fastA_*.npy；
     - 如果装了 onnxruntime，再在 CPU 上直接跑新 ONNX（grid_sampler 临时换成 onnxruntime 自带的 GridSample），
       与 PyTorch 输出对比。差异应在 0.01 像素以内，这一步通过就说明改图和换权重都正确。
  4) 打包 fastA_to_jetson.tar.gz（新 ONNX + 10 个参考输出），拷到 Jetson 用 jetson_fastA.sh 建引擎、测速。

在 SwitchStereo_PC 目录下运行：
    pip install onnxruntime      # 可选，用于第 3 步的 CPU 校验（强烈建议）
    P=/root/autodl-tmp/project/stereo-matching
    python make_fast_modelA.py --ref $P/checkpoints_ohem+spatial/model_best.pth --new $P/checkpoints_modelA/model_best.pth
"""
import argparse
import copy
import os
import re
import sys
import tarfile
from collections import defaultdict

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

sys.path.insert(0, os.getcwd())
import torchvision.models as _tvm  # noqa: E402
_orig = _tvm.resnet18
_tvm.resnet18 = lambda *a, **k: _orig(weights=None)   # 主干权重全部来自 checkpoint，不联网下载
from models.enhanced_stereonet import EnhancedStereoNet  # noqa: E402

SRC_NAME = "esnet_balanced_plugin_norange_trtfix_acfix.onnx"
OUT_NAME = "esnet_fastA_acfix.onnx"
H, W = 320, 640


# ----------------------------------------------------------------------------- 读取
def tload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def torch_sd(path):
    ck = tload(path)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    return {k.replace("module.", "", 1): v for k, v in sd.items() if torch.is_tensor(v)}


def np_sd(tsd):
    return {k: v.detach().float().numpy() for k, v in tsd.items() if v.is_floating_point()}


# ----------------------------------------------------------------------------- 改图
def stage_add(g, lvl):
    """找到 PRN 某一级 'disp + residual' 的 Add 节点，返回 (Add 节点, 输入视差张量名)。"""
    pat = re.compile(r"/progressive_refiner/%s(_\d+)?/res_net/res_net\.6/Conv" % lvl)
    convs = [n for n in g.node if n.op_type == "Conv" and
             (pat.search(n.name) or any(("stages.%s.res_net.6.weight" % lvl) in i for i in n.input))]
    if len(convs) != 1:
        sys.exit("!! 找不到 %s 级残差输出卷积（找到 %d 个），请把这句话发给我" % (lvl, len(convs)))
    r = convs[0].output[0]
    adds = [n for n in g.node if n.op_type == "Add" and r in n.input]
    if len(adds) != 1:
        sys.exit("!! %s 级残差卷积没有接到唯一的 Add 上（%d 个），请把这句话发给我" % (lvl, len(adds)))
    return adds[0], [i for i in adds[0].input if i != r][0]


def dce(g):
    prod = {o: n for n in g.node for o in n.output}
    keep, stack = set(), [o.name for o in g.output]
    while stack:
        n = prod.get(stack.pop())
        if n is None or id(n) in keep:
            continue
        keep.add(id(n))
        stack.extend(i for i in n.input if i)
    nodes = [n for n in g.node if id(n) in keep]
    del g.node[:]; g.node.extend(nodes)
    used = {i for n in g.node for i in n.input}
    inits = [t for t in g.initializer if t.name in used]
    del g.initializer[:]; g.initializer.extend(inits)
    produced = {o for n in g.node for o in n.output}
    vis = [v for v in g.value_info if v.name in produced]
    del g.value_info[:]; g.value_info.extend(vis)


def make_raw_graph(src):
    m = onnx.load(src); g = m.graph
    _, disp_p3 = stage_add(g, "p3")          # PRN 第一级的输入就是匹配模块输出的 1/8 视差
    out_name = g.output[0].name
    for n in g.node:
        for k, o in enumerate(n.output):
            if o == out_name:
                n.output[k] = out_name + "_unused"
    g.initializer.extend([
        numpy_helper.from_array(np.array([], dtype=np.float32), "fa_roi"),
        numpy_helper.from_array(np.array([1, 1, 8, 8], dtype=np.float32), "fa_scales"),
        numpy_helper.from_array(np.array(8.0, dtype=np.float32), "fa_eight"),
    ])
    g.node.extend([
        helper.make_node("Resize", [disp_p3, "fa_roi", "fa_scales"], ["fa_up"], name="fastA/Resize",
                         mode="linear", coordinate_transformation_mode="align_corners"),
        helper.make_node("Mul", ["fa_up", "fa_eight"], [out_name], name="fastA/Mul"),
    ])
    dce(g)
    return m


# ----------------------------------------------------------------------------- 换权重
def conv_for_bn(p, sd):
    cands = []
    mm = re.match(r"(.*)\.bn(\d+)$", p)
    if mm:
        cands.append(f"{mm.group(1)}.conv{mm.group(2)}")
    mm = re.match(r"(.*)\.(\d+)$", p)
    if mm:
        cands.append(f"{mm.group(1)}.{int(mm.group(2)) - 1}")
    for c in cands:
        w = sd.get(c + ".weight")
        if w is not None and w.ndim == 4 and w.shape[0] == sd[p + ".running_mean"].shape[0]:
            return c
    return None


def derived(sd):
    """权重在 ONNX 里可能出现的所有形式：原值，以及 Conv+BN 融合后的权重和偏置。"""
    d = {"raw:" + k: v for k, v in sd.items()}
    for k in sd:
        if not k.endswith(".running_mean"):
            continue
        p = k[: -len(".running_mean")]
        c = conv_for_bn(p, sd)
        if c is None:
            continue
        s = sd[p + ".weight"] / np.sqrt(sd[p + ".running_var"] + 1e-5)
        d["fusedW:" + c] = sd[c + ".weight"] * s.reshape(-1, 1, 1, 1)
        b = sd.get(c + ".bias", np.zeros_like(s))
        d["fusedB:" + c] = (b - sd[p + ".running_mean"]) * s + sd[p + ".bias"]
    return d


def float_tensors(g):
    out = []
    for t in g.initializer:
        a = numpy_helper.to_array(t)
        if a.dtype.kind == "f":
            out.append((t, a, t.name))
    for n in g.node:
        if n.op_type == "Constant":
            for at in n.attribute:
                if at.name == "value" and at.type == onnx.AttributeProto.TENSOR:
                    a = numpy_helper.to_array(at.t)
                    if a.dtype.kind == "f":
                        out.append((at.t, a, n.output[0]))
    return out


PASS_OPS = {"Unsqueeze", "Reshape", "Identity", "Cast", "Expand", "Squeeze", "Flatten"}


def module_path(node_name):
    """'/model/backbone/layer4/layer4.1/conv1/Conv' -> 'backbone.layer4.1.conv1'"""
    comps = [re.sub(r"_\d+$", "", c) for c in node_name.split("/")[1:-1] if c]
    if comps and comps[0] == "model":
        comps = comps[1:]
    path = []
    for c in comps:
        if path and c.startswith(path[-1] + "."):
            path[-1] = c
        else:
            path.append(c)
    return ".".join(path)


def structural_keys(name, consumers):
    """根据张量在图里被哪个节点、以什么身份使用，推出它对应的权重名。"""
    keys, stack, seen = set(), [(name, 0)], set()
    while stack:
        x, depth = stack.pop()
        for n, i in consumers.get(x, []):
            if n.op_type in PASS_OPS and i == 0 and depth < 4:
                if n.output[0] not in seen:
                    seen.add(n.output[0]); stack.append((n.output[0], depth + 1))
                continue
            path = module_path(n.name)
            if not path:
                continue
            if n.op_type == "Conv" and i in (1, 2):
                role = "weight" if i == 1 else "bias"
                keys.update({f"raw:{path}.{role}", ("fusedW:" if i == 1 else "fusedB:") + path})
            elif n.op_type == "Mul":
                keys.add(f"raw:{path}.weight")
            elif n.op_type == "Add":
                keys.add(f"raw:{path}.bias")
            elif n.op_type == "BatchNormalization" and 1 <= i <= 4:
                keys.add(f"raw:{path}." + ["weight", "bias", "running_mean", "running_var"][i - 1])
    return keys


def close(f, rf):
    return f.size == rf.size and np.allclose(f, rf.ravel().astype(np.float32), rtol=1e-4, atol=1e-6)


def swap_weights(m, ref_sd, new_sd):
    """按 ①同名 ②图结构（被哪个层、以什么身份使用）③数值唯一匹配 的顺序确定每个权重的来源，再换成 Model A 的值。"""
    ref_d, new_d = derived(ref_sd), derived(new_sd)
    has_bn = any(n.op_type == "BatchNormalization" for n in m.graph.node)
    consumers = defaultdict(list)
    for n in m.graph.node:
        for i, x in enumerate(n.input):
            consumers[x].append((n, i))
    by_n = defaultdict(list)
    for key, v in ref_d.items():
        f = v.ravel().astype(np.float32)
        if f.size > 1 and not np.all(f == f[0]):
            by_n[f.size].append((key, f))
    stats = defaultdict(int)
    unmatched, errors = [], []
    for t, a, name in float_tensors(m.graph):
        f = a.ravel().astype(np.float32)
        trivial = f.size <= 1 or bool(np.all(f == f[0]))
        key, how = None, None
        # ① 同名（未被折叠的权重保留 PyTorch 的名字，如 model.backbone.aspp.convs.0.0.weight）
        nm = name[6:] if name.startswith("model.") else name
        if "raw:" + nm in ref_d and close(f, ref_d["raw:" + nm]):
            key, how = "raw:" + nm, "同名"
        # ② 图结构
        if key is None:
            cands = [k for k in structural_keys(name, consumers) if k in ref_d and close(f, ref_d[k])]
            if not has_bn and any(k.startswith("fused") for k in cands):
                cands = [k for k in cands if k.startswith("fused")]   # 图里没有 BN 节点 = BN 已融合进 Conv
            news = {k: new_d.get(k) for k in cands}
            if cands and all(v is not None for v in news.values()) and \
                    all(np.array_equal(news[cands[0]], v) for v in news.values()):
                key, how = cands[0], "图结构"
            elif len(cands) > 1:
                errors.append(f"{name}{tuple(a.shape)}: 图结构给出多个来源 {cands}")
                continue
        # ③ 数值唯一匹配（非常数张量）
        if key is None and not trivial:
            hits = [k for k, rf in by_n.get(f.size, []) if close(f, rf)]
            news = [new_d.get(h) for h in hits]
            if len(hits) == 1 or (hits and all(v is not None for v in news) and
                                  all(np.array_equal(news[0], v) for v in news[1:])):
                key, how = hits[0], "数值"
            elif hits:
                errors.append(f"{name}{tuple(a.shape)}: 数值匹配到多个来源 {hits[:6]}")
                continue
        if key is None:
            if trivial:
                stats["导出常数（全同值），保留"] += 1
            else:
                unmatched.append((name, a.shape, float(f.min()), float(f.max())))
            continue
        nv = new_d.get(key)
        if nv is None or nv.size != f.size:
            errors.append(f"{name}{tuple(a.shape)}: Model A 里没有对应权重 {key}")
            continue
        t.CopyFrom(numpy_helper.from_array(nv.reshape(a.shape).astype(a.dtype), t.name))
        stats[f"{how}确定、已替换（{'Conv+BN 融合' if key.startswith('fused') else '原样'}）"] += 1
    return stats, unmatched, errors


def near_zero_report(ref_sd, new_sd):
    """顺便报告：最终模型里哪些模块的权重几乎全为 0（训练中被权重衰减压到 0 = 实际不起作用）。"""
    groups = defaultdict(lambda: [0.0, 0.0])
    for k, v in ref_sd.items():
        if k.endswith(("running_mean", "running_var")):
            continue
        parts = k.split(".")
        g = ".".join(parts[:3]) if parts[0] == "backbone" else ".".join(parts[:2])
        groups[g][0] = max(groups[g][0], float(np.abs(v).max()))
        if k in new_sd:
            groups[g][1] = max(groups[g][1], float(np.abs(new_sd[k]).max()))
    dead = [(g, a, b) for g, (a, b) in sorted(groups.items()) if a < 1e-4]
    if dead:
        print("  注意：以下模块在最终模型里权重几乎全为 0（最大绝对值 < 1e-4），说明训练中没有起作用：")
        for g, a, b in dead:
            print(f"    {g:<40} 最终模型 max|w| = {a:.2e}   Model A max|w| = {b:.2e}")


# ----------------------------------------------------------------------------- 校验
def to_ort_model(m):
    vm = copy.deepcopy(m)
    pad = {0: "zeros", 1: "border", 2: "reflection"}
    n_gs = 0
    for n in vm.graph.node:
        if n.op_type == "grid_sampler":
            at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            n.op_type, n.domain = "GridSample", "com.microsoft"
            del n.attribute[:]
            n.attribute.extend([
                helper.make_attribute("align_corners", int(at.get("align_corners", 0))),
                helper.make_attribute("mode", "bilinear"),
                helper.make_attribute("padding_mode", pad[int(at.get("padding_mode", 0))]),
            ])
            n_gs += 1
    if not any(o.domain == "com.microsoft" for o in vm.opset_import):
        vm.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return vm, n_gs


def d1_epe(pred, gt):
    v = (gt > 0) & (gt < 192)
    e = np.abs(pred[v] - gt[v])
    return e.mean(), ((e > 3) & (e > 0.05 * gt[v])).mean() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="部署 ONNX 当初导出用的权重（ohem+spatial）")
    ap.add_argument("--new", required=True, help="Model A 权重")
    ap.add_argument("--src", default=None, help=f"{SRC_NAME} 的路径（默认在当前目录下自动查找）")
    ap.add_argument("--cons", default="consistency_inputs")
    args = ap.parse_args()

    src = args.src
    if src is None:
        found = [os.path.join(r, f) for r, _, fs in os.walk(".") for f in fs if f == SRC_NAME and "to_jetson" not in r]
        if not found:
            sys.exit(f"!! 找不到 {SRC_NAME}，请用 --src 指定路径")
        src = found[0]
    out_path = os.path.join(os.path.dirname(src), OUT_NAME)
    print("=" * 90)
    print(f"源 ONNX：{src}")

    # 1) 改图
    m = make_raw_graph(src)
    g = m.graph
    gs = sum(1 for n in g.node if n.op_type == "grid_sampler")
    print(f"改图完成：节点 {len(g.node)} 个，grid_sampler {gs} 个（应为 24，全部在匹配模块），"
          f"PRN 相关 {sum(1 for n in g.node if 'progressive_refiner' in n.name)} 个（应为 0），"
          f"full_res_recon 相关 {sum(1 for n in g.node if 'full_res_recon' in n.name)} 个（应为 0）")

    # 2) 换权重
    ref_t, new_t = torch_sd(args.ref), torch_sd(args.new)
    ref_np, new_np = np_sd(ref_t), np_sd(new_t)
    stats, unmatched, errors = swap_weights(m, ref_np, new_np)
    print("换权重：" + "，".join(f"{k} {v} 个" for k, v in stats.items()))
    if unmatched:
        print(f"  在最终模型权重里找不到来源的张量 {len(unmatched)} 个（坐标网格等导出常数属正常，下面第 3 步的校验会确认）：")
        for name, shp, lo, hi in unmatched[:25]:
            print(f"    {name[:70]:<70} {str(tuple(shp)):<18} [{lo:.4g}, {hi:.4g}]")
    near_zero_report(ref_np, new_np)
    if errors:
        print("!! 以下权重无法确定如何替换，没有保存 ONNX。请把整段输出发给我：")
        for e in errors[:30]:
            print("   ", e)
        sys.exit(2)
    onnx.save(m, out_path)
    print(f"已保存 {out_path}")

    # 3) 校验
    print("=" * 90)
    names = [s.strip() for s in open(os.path.join(args.cons, "names.txt")) if s.strip()]
    model = EnhancedStereoNet(backbone_type="resnet18", max_disp=192, use_prn=False, use_gate=False)
    msg = model.load_state_dict(new_t, strict=False)
    print(f"PyTorch Model A：缺失 {len(msg.missing_keys)} 个，多余 {len(msg.unexpected_keys)} 个（都应为 0）")
    model.eval()
    data = {}
    with torch.no_grad():
        for s in names:
            l = np.fromfile(os.path.join(args.cons, f"left_{s}.bin"), np.float32).reshape(1, 3, H, W)
            r = np.fromfile(os.path.join(args.cons, f"right_{s}.bin"), np.float32).reshape(1, 3, H, W)
            pt = model(torch.from_numpy(l), torch.from_numpy(r))["disp_final"].squeeze().numpy()
            np.save(os.path.join(args.cons, f"pt_fastA_{s}.npy"), pt)
            data[s] = (l, r, pt, np.load(os.path.join(args.cons, f"gt_{s}.npy")).reshape(H, W))
    e, d = np.mean([d1_epe(v[2], v[3])[0] for v in data.values()]), np.mean([d1_epe(v[2], v[3])[1] for v in data.values()])
    print(f"已保存 {len(names)} 个 PyTorch 参考输出 pt_fastA_*.npy（这 10 张图上 EPE {e:.3f}，D1 {d:.2f}%）")

    try:
        import onnxruntime as ort
    except ImportError:
        ort = None
        print("没有安装 onnxruntime，跳过 CPU 校验（pip install onnxruntime 后重新运行本脚本即可）")
    if ort is not None:
        try:
            vm, n_gs = to_ort_model(m)
            sess = ort.InferenceSession(vm.SerializeToString(), providers=["CPUExecutionProvider"])
            ins = [i.name for i in sess.get_inputs()]
            diffs = []
            for s, (l, r, pt, gt) in data.items():
                feed = {ins[0]: l, ins[1]: r} if not {"left", "right"} <= set(ins) else {"left": l, "right": r}
                o = sess.run(None, feed)[0].reshape(H, W)
                diffs.append(np.abs(o - pt))
            dd = np.concatenate([x.ravel() for x in diffs])
            ok = dd.max() < 0.05
            print(f"onnxruntime CPU 校验（grid_sampler {n_gs} 个已临时替换）：|ONNX - PyTorch| 平均 {dd.mean():.5f}，"
                  f"最大 {dd.max():.5f} 像素 → {'通过 ✓' if ok else '不通过 ✗，请把整段输出发给我'}")
        except Exception as ex:  # noqa: BLE001
            print(f"onnxruntime 校验没能运行（{type(ex).__name__}: {str(ex)[:300]}），以 Jetson 上的一致性检查为准")

    # 4) 打包
    tgz = "fastA_to_jetson.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(out_path, arcname=OUT_NAME)
        for s in names:
            tf.add(os.path.join(args.cons, f"pt_fastA_{s}.npy"), arcname=f"consistency_inputs/pt_fastA_{s}.npy")
    print("=" * 90)
    print(f"已打包 {os.path.abspath(tgz)}（{os.path.getsize(tgz) / 1e6:.1f} MB）：下载后和 jetson_fastA.sh 一起拷到 Jetson 的 ~/Desktop/test_floder/")
    print("把以上全部输出发给我。")


if __name__ == "__main__":
    main()
