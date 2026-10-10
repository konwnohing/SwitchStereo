"""
检查一个（或几个）权重文件是不是部署 ONNX 里用的那一套权重。
ONNX 文件里本身就存着导出时的全部权重，所以拿它当"标准答案"来比对。

用法（电脑上，需要 torch 和 onnx）：
    python check_ckpt_vs_onnx.py esnet_accurate_plugin_norange_trtfix.onnx  model_best.pth  [另一个.pth ...]

判断标准：RCM / PRN / Gate / FPN 等层的卷积权重如果绝大多数（>80%）能在 ONNX 里找到完全相同的数值，
就是同一套权重；如果几乎找不到（<10%），就是不同的权重。
（ResNet 主干里的卷积导出时会和 BN 合并，数值会变，所以不参与比对。）
"""
import sys
from collections import defaultdict

import numpy as np
import onnx
import torch
from onnx import numpy_helper

SKIP_PREFIX = (
    "backbone.conv1", "backbone.layer",                 # 与 BN 融合的 ResNet 卷积，数值会变
    "progressive_refiner.stages.p5", "progressive_refiner.stages.p4",          # PRN 从 p3 开始，
    "progressive_refiner.upsamplers.p5", "progressive_refiner.upsamplers.p4",  # 这些层推理时不用，
    "progressive_refiner.upsamplers.p3",                                       # 不会出现在 ONNX 里
)


def torch_load_cpu(path):
    """兼容新旧版 PyTorch：新版（2.6+）默认 weights_only=True，读带其他信息的 checkpoint 会报错。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # 旧版没有 weights_only 参数
        return torch.load(path, map_location="cpu")


def load_sd(path):
    ckpt = torch_load_cpu(path)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    return {k.replace("module.", "", 1): v for k, v in sd.items() if torch.is_tensor(v)}


def main():
    only = None
    if "--only" in sys.argv:  # 只比对指定前缀的层，例如 Fast 模型里没有 PRN/Gate：--only rcm_matcher,backbone.aspp
        i = sys.argv.index("--only")
        only = tuple(sys.argv[i + 1].split(","))
        del sys.argv[i:i + 2]
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    model = onnx.load(sys.argv[1])
    by_shape = defaultdict(list)
    tensors = list(model.graph.initializer)
    for n in model.graph.node:  # 有些权重以 Constant 节点形式存放
        if n.op_type == "Constant":
            tensors += [a.t for a in n.attribute if a.name == "value"]
    for t in tensors:
        a = numpy_helper.to_array(t)
        if a.ndim == 4:
            by_shape[tuple(a.shape)].append(a.astype(np.float32))
    print(f"ONNX：{sys.argv[1]}，4 维权重 {sum(len(v) for v in by_shape.values())} 个")

    all_same = True
    for ck in sys.argv[2:]:
        sd = load_sd(ck)
        keys = [k for k, v in sd.items() if v.ndim == 4 and k.endswith("weight") and not k.startswith(SKIP_PREFIX)
                and (only is None or k.startswith(only))]
        hit = 0
        for k in keys:
            w = sd[k].float().numpy()
            if any(np.allclose(w, a, rtol=0, atol=1e-6) for a in by_shape.get(tuple(w.shape), [])):
                hit += 1
        ratio = hit / max(len(keys), 1) * 100
        verdict = "同一套权重" if ratio > 80 else ("不是同一套权重" if ratio < 10 else "部分一致，请把结果发给我")
        print(f"{ck}：比对 {len(keys)} 个卷积权重，完全一致 {hit} 个（{ratio:.1f}%）→ {verdict}")
        all_same &= ratio > 80
    sys.exit(0 if all_same else 2)  # 一键脚本据此判断是否继续


if __name__ == "__main__":
    main()
