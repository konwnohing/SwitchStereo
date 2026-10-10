"""
比较同一套权重下几种“部署变体”的精度和部署口径 MACs（只推理，有卡约 2–5 分钟）。

变体（都用同一个 checkpoint，不需要重新训练）：
  full        : 完整网络（Accurate：匹配 + PRN + Gate）
  no_gate     : 去掉 Gate（Balanced）
  pruned      : 去掉 Gate，并去掉 PRN 中 p1、full_res 两级的残差分支（这两级只保留凸上采样）
  convex_only : 去掉 Gate 和 PRN 所有残差分支，只保留各级凸上采样（候选的新 Fast 模式）
  bilinear    : 现在的 Fast 模式（1/8 视差直接双线性放大）
评估协议与论文一致：KITTI 2015 的 000121–000199 共 79 对；(A) 384x1152 中心裁剪；(B) 全图（补零到 384x1248）。

在 SwitchStereo_PC 目录下运行：
    P=/root/autodl-tmp/project/stereo-matching
    python eval_variants.py --kitti_root /root/autodl-tmp/kitti_dataset \
        --ckpts $P/checkpoints_ohem+spatial/model_best.pth $P/checkpoints_modelbaseline/model_best.pth
（MACs 只对第一个 checkpoint 统计，结构相同时结果通用。）
"""
import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.getcwd())
import torchvision.models as _tvm  # noqa: E402
_orig = _tvm.resnet18
_tvm.resnet18 = lambda *a, **k: _orig(weights=None)
from models.enhanced_stereonet import EnhancedStereoNet  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
VARIANTS = {
    "full": (set(), "disp_final"),
    "no_gate": (set(), "disp_init"),
    "pruned": ({"p1", "full_res"}, "disp_init"),
    "convex_only": ({"p3", "p2", "p1", "full_res"}, "disp_init"),
    "bilinear": (set(), "p3"),
}
LEAF = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.GroupNorm)
MACT = (nn.Conv2d, nn.Linear)
GROUPS = [("backbone.aspp", "ASPP"), ("backbone.fpn", "FPN"), ("backbone.p1_enhance", "p1_enh"),
          ("backbone.full_res_recon", "full_res"), ("backbone.", "ResNet"), ("rcm_matcher", "RCM"),
          ("progressive_refiner.upsamplers", "PRN-up"), ("progressive_refiner.stages", "PRN-res"), ("gate_refiner", "Gate")]


def group_of(n):
    return next((g for p, g in GROUPS if n.startswith(p)), "other")


def load_sd(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


class Net:
    """包装模型：用 forward hook 按变体跳过 PRN 残差分支，并支持逐层 MACs 统计 / 扰动判断。"""

    def __init__(self, path, dev):
        self.m = EnhancedStereoNet(backbone_type="resnet18", max_disp=192, use_prn=True, use_gate=True)
        missing, _ = self.m.load_state_dict(load_sd(path), strict=False)
        self.missing = len(missing)
        self.m = self.m.to(dev).eval()
        self.skip = set()
        for name, st in self.m.progressive_refiner.stages.items():
            st.register_forward_hook(self._stage_hook(name))
        self.mode, self.idx, self.target, self.calls = "off", 0, -1, []
        for name, mod in self.m.named_modules():
            if isinstance(mod, LEAF):
                mod.register_forward_hook(self._leaf_hook(name))

    def _stage_hook(self, name):
        return lambda mod, inp, out: inp[0] if name in self.skip else None

    def _leaf_hook(self, name):
        def h(mod, inp, out):
            i = self.idx; self.idx += 1
            if self.mode == "record":
                macs = 0
                if isinstance(mod, nn.Conv2d):
                    macs = out.numel() * (mod.in_channels // mod.groups) * math.prod(mod.kernel_size)
                elif isinstance(mod, nn.Linear):
                    macs = out.numel() * mod.in_features
                self.calls.append((name, macs, sum(p.numel() for p in mod.parameters(recurse=False))))
            elif self.mode == "perturb" and i == self.target:
                g = torch.Generator(device=out.device).manual_seed(1234)
                return out + torch.randn(out.shape, generator=g, device=out.device, dtype=out.dtype) * (float(out.detach().float().std()) + 1.0) * 10
            return None
        return h

    @torch.no_grad()
    def run(self, l, r, variant, mode="off", target=-1):
        self.skip, key = VARIANTS[variant]
        self.mode, self.idx, self.target = mode, 0, target
        if mode == "record":
            self.calls = []
        out = self.m(l, r)
        self.mode = "off"
        if key == "p3":
            p3 = out["all_disparities"]["p3"]
            return F.interpolate(p3, size=l.shape[-2:], mode="bilinear", align_corners=True) * (l.shape[-1] / p3.shape[-1])
        return out[key]


def prep_crop(a, CH=384, CW=1152):
    h, w = a.shape[:2]
    a = np.pad(a, [(0, max(0, CH - h)), (0, max(0, CW - w))] + ([(0, 0)] if a.ndim == 3 else []))
    h, w = a.shape[:2]
    t, l = int(round((h - CH) / 2.0)), int(round((w - CW) / 2.0))
    return a[t:t + CH, l:l + CW]


def prep_full(a, H=384, W=1248):
    h, w = a.shape[:2]
    return np.pad(a, [(0, H - h), (0, W - w)] + ([(0, 0)] if a.ndim == 3 else []))


def to_t(x, dev):
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--val_start", type=int, default=121)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), help="只评估指定变体")
    ap.add_argument("--no_macs", action="store_true", help="不统计 MACs（只要精度时加上，更快）")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join(args.kitti_root, "training")
    names = sorted(n for n in os.listdir(os.path.join(root, "image_2")) if n.endswith("_10.png") and int(n[:6]) >= args.val_start)
    rd = lambda sub, n: np.asarray(Image.open(os.path.join(root, sub, n)).convert("RGB"), np.float32) / 255.0
    data = []
    for n in names:
        l, r = rd("image_2", n), rd("image_3", n)
        gt = np.asarray(Image.open(os.path.join(root, "disp_occ_0", n)), np.float32) / 256.0
        data.append((l, r, gt))
    print(f"验证集 {len(names)} 对：{names[0]} ~ {names[-1]}，设备 {dev}")

    acc_rows, mac_rows = [], []
    for ci, path in enumerate(args.ckpts):
        net = Net(path, dev)
        tag = os.path.basename(os.path.dirname(path))
        print(f"\n[{tag}] missing={net.missing}")
        for var in args.variants:
            res = {}
            for prot in ("crop", "full"):
                es, ds, eacc, bacc, nacc = [], [], 0.0, 0, 0
                for l, r, gt in data:
                    h, w = gt.shape
                    if prot == "crop":
                        L_, R_, G_ = to_t(prep_crop(l), dev), to_t(prep_crop(r), dev), prep_crop(gt)
                        d = net.run(L_, R_, var).squeeze().float().cpu().numpy()
                    else:
                        L_, R_, G_ = to_t(prep_full(l), dev), to_t(prep_full(r), dev), gt
                        d = net.run(L_, R_, var).squeeze().float().cpu().numpy()[:h, :w]
                    v = (G_ > 0) & (G_ < 192)
                    e = np.abs(d[v] - G_[v]); b = (e > 3) & (e > 0.05 * G_[v])
                    es.append(e.mean()); ds.append(b.mean() * 100)
                    eacc += e.sum(); bacc += b.sum(); nacc += v.sum()
                res[prot] = (np.mean(es), np.mean(ds), bacc / nacc * 100)
            acc_rows.append((tag, var, *res["crop"], *res["full"]))
            print(f"  {var:<12} 完成")

        mac_rows = mac_rows if ci else []
        if ci == 0 and not args.no_macs:  # 部署口径 MACs：判断每次层调用对该变体输出是否有贡献
            l0, r0, _ = data[-1]
            ld = [to_t(prep_crop(l0, 320, 640), dev), to_t(prep_crop(r0, 320, 640), dev)]
            for var in args.variants:
                base = net.run(*ld, var, mode="record").float()
                calls = list(net.calls)
                used = []
                for i in range(len(calls)):
                    diff = float((net.run(*ld, var, mode="perturb", target=i).float() - base).abs().max())
                    used.append(not (diff <= 1e-3))
                row = [var]
                for (H, W) in [(320, 640), (384, 1248)]:
                    zz = [torch.zeros(1, 3, H, W, device=dev)] * 2
                    net.run(*zz, var, mode="record")
                    row.append(sum(m for (_, m, _), u in zip(net.calls, used) if u))
                    if (H, W) == (320, 640):
                        pg = defaultdict(float)
                        for (n, m, _), u in zip(net.calls, used):
                            if u:
                                pg[group_of(n)] += m
                        row.append(dict(pg))
                used_mods = {n for (n, _, _), u in zip(calls, used) if u}
                row.append(sum(p for n, _, p in {(n, 0, p) for (n, _, p) in calls} if n in used_mods))
                mac_rows.append(row)
        del net
        if dev == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 104)
    print("表 1：精度（79 对验证集）")
    print(f"{'checkpoint':<26}{'variant':<13}| (A) 384x1152: EPE  D1逐图  D1像素 | (B) 全图: EPE  D1逐图  D1像素")
    print("-" * 104)
    for tag, var, e1, d1, p1, e2, d2, p2 in acc_rows:
        print(f"{tag:<26}{var:<13}|        {e1:6.3f} {d1:6.2f} {p1:6.2f}  |      {e2:6.3f} {d2:6.2f} {p2:6.2f}")
    print("=" * 104)
    print("表 2：部署口径 MACs（只计对输出有贡献的卷积）与参数量")
    for var, m320, pg, m384, params in mac_rows:
        detail = "  ".join(f"{k}:{v / 1e9:.2f}" for k, v in sorted(pg.items(), key=lambda x: -x[1]))
        print(f"  {var:<12} 320x640: {m320 / 1e9:7.2f} G   384x1248: {m384 / 1e9:7.2f} G   参数(含归一化层): {params / 1e6:6.3f} M")
        print(f"               320x640 各模块(G): {detail}")
    print("=" * 104)
    print("把以上两张表整体复制给我即可。")


if __name__ == "__main__":
    main()
