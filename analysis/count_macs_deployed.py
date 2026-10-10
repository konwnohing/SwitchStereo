"""
统计 SwitchStereo 三种模式"实际部署时"的参数量与 MACs（在训练服务器上运行，只做前向，不训练）。

为什么需要：
  PyTorch 里 Fast 模式的 forward 仍然会计算 full_res_recon、p1_enhance 等分支（结果没被用到），
  所以用 thop 之类工具直接统计会把这些"死代码"也算进去；而 TensorRT 构建引擎时会把它们删掉
  （Jetson 逐层剖析里 Fast 的 full_res_recon / p1_enhance 耗时为 0）。论文表 7 的 Fast MACs 因此偏大。

做法：
  对每个卷积 / 全连接 / 归一化层的每一次调用，单独给它的输出加大噪声，再看最终视差 disp_final 是否改变。
  不改变 = 这次计算对输出没有贡献（部署时会被删掉）；改变 = 部署时会保留。
  MACs 只统计卷积和全连接（与 thop 口径一致），相关性计算、插值、grid_sample 不计入。

在项目根目录（stereo-matching/）下运行：

    python count_macs_deployed.py \
        --ckpt checkpoints_ohem+spatial/model_best.pth \
        --kitti_root /root/autodl-tmp/kitti_dataset

没有 GPU 也能跑（自动用 CPU），只是慢一些；CPU 上会自动用小尺寸做判断以节省时间。
运行完把最后打印的几张表整体复制给我。
"""
import argparse
import math
import os
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())
from models.enhanced_stereonet import EnhancedStereoNet  # noqa: E402

import torchvision.models as _tvm  # noqa: E402
_orig_resnet18 = _tvm.resnet18
# 主干权重全部来自 checkpoint，这里不再联网下载 ImageNet 预训练权重
_tvm.resnet18 = lambda *a, **k: _orig_resnet18(weights=None)


def torch_load_cpu(path):
    """兼容新旧版 PyTorch：新版（2.6+）默认 weights_only=True，读带其他信息的 checkpoint 会报错。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # 旧版没有 weights_only 参数
        return torch.load(path, map_location="cpu")

MODES = OrderedDict(
    fast=dict(use_prn=False, use_gate=False),
    balanced=dict(use_prn=True, use_gate=False),
    accurate=dict(use_prn=True, use_gate=True),
)
LEAF_TYPES = (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear,
              nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm)
MAC_TYPES = (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
GROUPS = [  # (名称前缀, 分组名)；按顺序匹配第一个
    ("backbone.aspp", "ASPP"),
    ("backbone.fpn", "FPN"),
    ("backbone.p1_enhance", "p1_enhance"),
    ("backbone.full_res_recon", "full_res_recon"),
    ("backbone.", "ResNet-18"),
    ("rcm_matcher", "RCM"),
    ("progressive_refiner", "PRN"),
    ("gate_refiner", "Gate"),
]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def group_of(name):
    for prefix, g in GROUPS:
        if name.startswith(prefix):
            return g
    return "other"


def layer_macs(m, inp, out):
    if isinstance(m, nn.Linear):
        return out.numel() * m.in_features
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        return out.numel() * (m.in_channels // m.groups) * math.prod(m.kernel_size)
    if isinstance(m, nn.ConvTranspose2d):
        return inp[0].numel() * (m.out_channels // m.groups) * math.prod(m.kernel_size)
    return 0


class Tracker:
    """给每个叶子层挂 forward hook：record 模式记录每次调用的 MACs；perturb 模式给指定调用的输出加噪声。"""

    def __init__(self, model):
        self.mode, self.idx, self.target = "off", 0, -1
        self.calls = []  # [(层名, MACs)]
        for name, m in model.named_modules():
            if isinstance(m, LEAF_TYPES):
                m.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(m, inp, out):
            if not torch.is_tensor(out):
                return None
            i = self.idx
            self.idx += 1
            if self.mode == "record":
                self.calls.append((name, layer_macs(m, inp, out) if isinstance(m, MAC_TYPES) else 0))
            elif self.mode == "perturb" and i == self.target:
                g = torch.Generator(device=out.device).manual_seed(1234)
                noise = torch.randn(out.shape, generator=g, device=out.device, dtype=out.dtype)
                scale = float(out.detach().float().std()) + 1.0
                return out + noise * scale * 10.0
            return None
        return hook

    @torch.no_grad()
    def run(self, model, l, r, mode="off", target=-1):
        self.mode, self.idx, self.target = mode, 0, target
        if mode == "record":
            self.calls = []
        out = model(l, r)
        d = out["disp_final"] if isinstance(out, dict) else out
        self.mode = "off"
        return d.float()


def load_model(ckpt_path, mode, device):
    model = EnhancedStereoNet(backbone_type="resnet18", max_disp=192, **MODES[mode])
    if ckpt_path:
        ckpt = torch_load_cpu(ckpt_path)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, _ = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  !! [{mode}] 缺失权重 {len(missing)} 个，例如 {missing[:3]}")
    return model.to(device).eval()


def real_pair(kitti_root, H, W, device):
    """取 KITTI 训练集最后一张图的中心区域（只用于判断哪些层有贡献，不涉及精度）。"""
    from PIL import Image
    img_dir = os.path.join(kitti_root, "training", "image_2")
    name = sorted(n for n in os.listdir(img_dir) if n.endswith("_10.png"))[-1]
    out = []
    for sub in ("image_2", "image_3"):
        a = np.asarray(Image.open(os.path.join(kitti_root, "training", sub, name)).convert("RGB"), np.float32) / 255.0
        h, w = a.shape[:2]
        y, x = max((h - H) // 2, 0), max((w - W) // 2, 0)
        a = a[y:y + H, x:x + W]
        a = np.pad(a, [(0, H - a.shape[0]), (0, W - a.shape[1]), (0, 0)])
        a = (a - MEAN) / STD
        out.append(torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(device))
    return out


def rand_pair(H, W, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 3, H, W, generator=g).to(device) for _ in range(2)]


def fmt_g(x):
    return f"{x / 1e9:8.2f}G"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--kitti_root", default="")
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--detect_res", nargs=2, type=int, default=None,
                    help="判断层是否有贡献时用的输入尺寸，默认 GPU 用 320 640、CPU 用 160 320")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    resolutions = [(320, 640), (384, 1248)]
    summary, breakdown, dead_lists = [], {}, {}

    for mode in args.modes:
        print(f"\n===== {mode} =====")
        model = load_model(args.ckpt, mode, args.device)
        tr = Tracker(model)

        # 1) 判断每次调用是否对输出有贡献（结构与分辨率无关，CPU 上用小尺寸以节省时间）
        dh, dw = args.detect_res if args.detect_res else ((320, 640) if args.device.startswith("cuda") else (160, 320))
        if mode == args.modes[0]:
            print(f"  设备 {args.device}，判断用的输入尺寸 {dh}x{dw}")
        l, r = real_pair(args.kitti_root, dh, dw, args.device) if args.kitti_root else rand_pair(dh, dw, args.device)
        base = tr.run(model, l, r, mode="record")
        calls_det = list(tr.calls)
        floor = float((tr.run(model, l, r) - base).abs().max())  # 重复运行的数值抖动
        thr = max(1e-3, floor * 100)
        used = []
        for i in range(len(calls_det)):
            diff = float((tr.run(model, l, r, mode="perturb", target=i) - base).abs().max())
            used.append(not (diff <= thr))  # NaN 也算"有贡献"
            if (i + 1) % 50 == 0:
                print(f"  已检查 {i + 1}/{len(calls_det)} 次层调用")
        print(f"  共 {len(calls_det)} 次层调用，其中对输出无贡献的 {used.count(False)} 次（阈值 {thr:.1e}）")

        # 2) 参数量：模型全部参数 vs 部署时保留的层的参数
        name2mod = dict(model.named_modules())
        used_names = {n for (n, _), u in zip(calls_det, used) if u}
        p_total = sum(p.numel() for p in model.parameters())
        p_leaf_used = sum(p.numel() for n in used_names for p in name2mod[n].parameters(recurse=False))
        p_other = sum(p.numel() for n, m in model.named_modules() if not isinstance(m, LEAF_TYPES)
                      for p in m.parameters(recurse=False))
        p_dep = p_leaf_used + p_other

        # 3) 各分辨率下的 MACs（调用顺序与分辨率无关，直接复用上面的判断结果）
        row = [mode, p_total / 1e6, p_dep / 1e6]
        for (H, W) in resolutions:
            tr.run(model, *rand_pair(H, W, args.device), mode="record")
            assert len(tr.calls) == len(calls_det), "不同分辨率下层调用次数不一致"
            all_macs = sum(m for _, m in tr.calls)
            dep_macs = sum(m for (_, m), u in zip(tr.calls, used) if u)
            row += [all_macs, dep_macs]
            per_g = defaultdict(lambda: [0, 0])
            for (n, m), u in zip(tr.calls, used):
                per_g[group_of(n)][0] += m
                per_g[group_of(n)][1] += m if u else 0
            breakdown[(mode, H, W)] = per_g
        summary.append(row)

        dead = defaultdict(int)
        for (n, _), u in zip(calls_det, used):
            if not u:
                dead[n] += 1
        dead_lists[mode] = dead
        del model, tr
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---------------- 输出 ----------------
    print("\n" + "=" * 100)
    print("表 A：参数量与 MACs（all = PyTorch 前向实际执行的全部；deployed = 去掉对输出无贡献的计算，对应 TensorRT 引擎）")
    print("-" * 100)
    print(f"{'mode':<9}| Params(model) Params(deployed) | 320x640: MACs(all)  MACs(deployed) | 384x1248: MACs(all)  MACs(deployed)")
    for mode, pt, pd, a1, d1, a2, d2 in summary:
        print(f"{mode:<9}|   {pt:8.3f}M     {pd:8.3f}M     |      {fmt_g(a1)}     {fmt_g(d1)}   |       {fmt_g(a2)}     {fmt_g(d2)}")
    for (H, W) in resolutions:
        print("=" * 100)
        print(f"表 B：各模块 deployed MACs @ {H}x{W}（括号内为 PyTorch 实际执行的 all）")
        print("-" * 100)
        names = [g for _, g in GROUPS] + ["other"]
        print(f"{'mode':<9}| " + " | ".join(f"{g:>16}" for g in names))
        for mode in [s[0] for s in summary]:
            pg = breakdown[(mode, H, W)]
            cells = []
            for g in names:
                a, d = pg.get(g, [0, 0])
                cells.append(f"{d / 1e9:6.2f}({a / 1e9:6.2f})".rjust(16))
            print(f"{mode:<9}| " + " | ".join(cells))
    print("=" * 100)
    print("表 C：对输出无贡献的层（层名: 调用次数）——用于和 Jetson 逐层剖析核对")
    for mode, dead in dead_lists.items():
        by_g = defaultdict(list)
        for n, c in dead.items():
            by_g[group_of(n)].append(f"{n}x{c}")
        print(f"[{mode}] " + ("无" if not dead else ""))
        for g, lst in by_g.items():
            print(f"   {g}: {len(lst)} 个层  例如 {', '.join(lst[:4])}")
    print("=" * 100)
    print("把以上表 A、B、C 整体复制给我即可。")


if __name__ == "__main__":
    main()
