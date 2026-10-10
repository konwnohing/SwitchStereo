"""
用现有权重、统一协议，一次性重算论文表 2–7、表 11 需要的全部精度数字（只推理，有卡约 3–6 分钟）。

统一协议：KITTI 2015 训练集中 000121–000199 共 79 对（所有 KITTI 模型训练时都没见过），
  (A) 384x1152 中心裁剪（与训练时验证完全一致）
  (B) 全图（上方、右侧补零到 32 的倍数，与仓库 tests/ 脚本一致）
  边界指标：在全图上，取左图 Sobel 梯度最强的 15% 像素并膨胀 3 像素（与 tests/boundary_region_analysis_six.py 一致）
  深度指标：分析模型在全图上按真值深度分 0–20 / 20–40 / 40–80 m（需要 training/calib_cam_to_cam/，没有则跳过）

在 SwitchStereo_PC 目录下运行（默认会评估下面列出的所有模型）：
    python reproduce_tables.py --proj /root/autodl-tmp/project/stereo-matching --kitti_root /root/autodl-tmp/kitti_dataset
如有 OHEM 比例实验（表 5）的权重，可追加：  --extra 路径:变体:名称  路径:变体:名称 ...
只评估 --extra 里的模型：加 --only_extra；换验证起点（例如只用 KITTI 2015 训练的模型用 000160–000199）：--val_start 160
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2
except ImportError:
    sys.exit("需要 opencv：pip install opencv-python-headless")

sys.path.insert(0, os.getcwd())
import torchvision.models as _tvm  # noqa: E402
_orig = _tvm.resnet18
_tvm.resnet18 = lambda *a, **k: _orig(weights=None)
from models.enhanced_stereonet import EnhancedStereoNet  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ALL = {"p3", "p2", "p1", "full_res"}
VARIANTS = {   # 名称: (跳过残差的 PRN 级别, 取哪个输出)
    "full": (set(), "disp_final"),
    "no_gate": (set(), "disp_init"),
    "pruned": ({"p1", "full_res"}, "disp_init"),
    "convex_only": (ALL, "disp_init"),
    "fast_v2": (set(), "p3"),          # 1/8 一级修正 + 双线性（新 Fast）
    "raw_bilinear": (ALL, "p3"),       # 未修正的 1/8 视差 + 双线性（原 Fast）
}


def load_sd(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


class Net:
    def __init__(self, path, dev):
        self.m = EnhancedStereoNet(backbone_type="resnet18", max_disp=192, use_prn=True, use_gate=True)
        missing, _ = self.m.load_state_dict(load_sd(path), strict=False)
        self.has_gate = not any(k.startswith("gate_refiner") for k in missing)
        self.missing = [k for k in missing if not k.startswith("gate_refiner")]
        self.m = self.m.to(dev).eval()
        self.skip = set()
        for name, st in self.m.progressive_refiner.stages.items():
            st.register_forward_hook(lambda mod, i, o, name=name: i[0] if name in self.skip else None)

    @torch.no_grad()
    def __call__(self, l, r, variant):
        self.skip, key = VARIANTS[variant]
        out = self.m(l, r)
        if key == "p3":
            p3 = out["all_disparities"]["p3"]
            return F.interpolate(p3, size=l.shape[-2:], mode="bilinear", align_corners=True) * (l.shape[-1] / p3.shape[-1])
        return out[key]


def to_t(x, dev):
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(dev)


def crop(a, CH=384, CW=1152):
    h, w = a.shape[:2]
    a = np.pad(a, [(0, max(0, CH - h)), (0, max(0, CW - w))] + ([(0, 0)] if a.ndim == 3 else []))
    h, w = a.shape[:2]
    t, l = int(round((h - CH) / 2.0)), int(round((w - CW) / 2.0))
    return a[t:t + CH, l:l + CW]


def pad32(a):
    h, w = a.shape[:2]
    tp, rp = (-h) % 32, (-w) % 32
    return np.pad(a, [(tp, 0), (0, rp)] + ([(0, 0)] if a.ndim == 3 else [])), tp


def edge_mask(img):
    g = cv2.GaussianBlur(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), (3, 3), 0)
    mag = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) ** 2)
    v = mag[mag > 0]
    e = mag >= np.percentile(v, 85) if v.size else np.zeros_like(mag, bool)
    return cv2.dilate(e.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool)


def calib(root, name):
    p = os.path.join(root, "training", "calib_cam_to_cam", name[:6] + ".txt")
    if not os.path.exists(p):
        return None
    P = {}
    for line in open(p):
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                P[k.strip()] = np.array([float(x) for x in v.split()])
            except ValueError:
                pass
    p2 = P.get("P_rect_02", P.get("P2")); p3 = P.get("P_rect_03", P.get("P3"))
    if p2 is None or p3 is None:
        return None
    f = p2[0]
    return f, abs((-p3[3] / p3[0]) - (-p2[3] / p2[0]))


class Acc:
    def __init__(self):
        self.e, self.b, self.n, self.img_d1, self.img_epe = 0.0, 0, 0, [], []

    def add(self, pred, gt, mask=None):
        v = (gt > 0) & (gt < 192)
        if mask is not None:
            v &= mask
        if not v.any():
            return
        e = np.abs(pred[v] - gt[v]); bad = (e > 3) & (e > 0.05 * gt[v])
        self.e += e.sum(); self.b += bad.sum(); self.n += v.sum()
        self.img_d1.append(bad.mean() * 100); self.img_epe.append(e.mean())

    def res(self):
        return np.mean(self.img_epe), np.mean(self.img_d1), self.e / self.n, self.b / self.n * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="路径:变体:名称")
    ap.add_argument("--only_extra", action="store_true", help="只评估 --extra 里的模型")
    ap.add_argument("--val_start", type=int, default=121, help="验证集起始编号（默认 121，即 79 对）")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = lambda d: os.path.join(args.proj, d, "model_best.pth")
    runs = [
        # 训练层面的消融（各自单独训练的模型）
        (P("checkpoints_modelrpn"), "no_gate", "T: PRN, no Gate, L1 (modelrpn)"),
        (P("checkpoints_baseline"), "full", "T: PRN+Gate, L1 (baseline)"),
        (P("checkpoints_ohem"), "full", "T: PRN+Gate, OHEM (ohem)"),
        (P("checkpoints_ohem+spatial"), "full", "T: PRN+Gate, OHEM+edge (final)"),
        # 同一套权重（最终模型）在推理时开关各部件
        (P("checkpoints_ohem+spatial"), "raw_bilinear", "I: D_p3 + bilinear (old Fast)"),
        (P("checkpoints_ohem+spatial"), "fast_v2", "I: + 1/8 refinement (new Fast)"),
        (P("checkpoints_ohem+spatial"), "convex_only", "I: D_p3 + convex up only"),
        (P("checkpoints_ohem+spatial"), "pruned", "I: PRN pruned (new Balanced)"),
        (P("checkpoints_ohem+spatial"), "no_gate", "I: full PRN, no Gate"),
        (P("checkpoints_ohem+spatial"), "full", "I: full PRN + Gate"),
        # 零样本
        (P("checkpoints_sceneflow_pretrain_1"), "full", "Z: Scene Flow only (zero-shot)"),
    ]
    extra = [tuple(x.rsplit(":", 2)) for x in args.extra]
    runs = extra if args.only_extra else runs + extra

    root = os.path.join(args.kitti_root, "training")
    names = sorted(n for n in os.listdir(os.path.join(root, "image_2")) if n.endswith("_10.png") and int(n[:6]) >= args.val_start)
    rd = lambda sub, n: np.asarray(Image.open(os.path.join(root, sub, n)).convert("RGB"), np.float32) / 255.0
    data = [(n, rd("image_2", n), rd("image_3", n), np.asarray(Image.open(os.path.join(root, "disp_occ_0", n)), np.float32) / 256.0) for n in names]
    masks = [edge_mask(l) for _, l, _, _ in data]
    print(f"验证集 {len(names)} 对：{names[0]} ~ {names[-1]}，设备 {dev}")

    rows, depth = [], None
    cache = {}
    for path, var, label in runs:
        if not os.path.exists(path):
            print(f"  跳过（找不到）：{path}")
            continue
        if path not in cache:
            cache.clear(); cache[path] = Net(path, dev)
        net = cache[path]
        if var == "full" and not net.has_gate:
            var = "no_gate"
        A, B, Bd = Acc(), Acc(), Acc()
        dep = {k: [] for k in ("Near (0-20 m)", "Middle (20-40 m)", "Far (40-80 m)", "All (0-80 m)")}
        want_depth = label.startswith("I: PRN pruned")
        for (n, l, r, gt), em in zip(data, masks):
            d = net(to_t(crop(l), dev), to_t(crop(r), dev), var).squeeze().float().cpu().numpy()
            A.add(d, crop(gt))
            (lp, tp), (rp, _) = pad32(l), pad32(r)
            h, w = gt.shape
            d = net(to_t(lp, dev), to_t(rp, dev), var).squeeze().float().cpu().numpy()[tp:tp + h, :w]
            B.add(d, gt); Bd.add(d, gt, em)
            if want_depth:
                c = calib(args.kitti_root, n)
                if c is None:
                    continue
                f, base = c
                v = (gt > 0) & (d > 0.1)
                gd, pd = f * base / np.maximum(gt, 1e-6), f * base / np.maximum(d, 1e-6)
                v &= gd <= 80
                for k, (lo, hi) in zip(dep, [(0, 20), (20, 40), (40, 80), (0, 80)]):
                    s = v & (gd >= lo) & (gd < hi)
                    if s.any():
                        dep[k].append((pd[s] - gd[s], gd[s], d[s] - gt[s]))
        rows.append((label, var, len(net.missing), *A.res(), *B.res(), *Bd.res()))
        print(f"  完成：{label}")
        if want_depth:
            depth = dep

    print("\n" + "=" * 140)
    print(f"表 1：精度（{len(names)} 对）。EPE/D1 为逐图平均（与训练时验证一致），括号内为按像素累计")
    print(f"{'run':<38}{'variant':<13}{'miss':>5} | {'(A) 384x1152 EPE':>17} {'D1':>14} | {'(B) 全图 EPE':>13} {'D1':>14} | {'边界 EPE':>10} {'边界 D1':>14}")
    print("-" * 140)
    for label, var, ms, ea, da, eap, dap, eb, db, ebp, dbp, ee, de_, eep, dep_ in rows:
        print(f"{label:<38}{var:<13}{ms:>5} | {ea:9.3f}({eap:.3f}) {da:6.2f}({dap:5.2f}) | {eb:7.3f}({ebp:.3f}) {db:6.2f}({dbp:5.2f}) | {ee:5.3f}({eep:.3f}) {de_:6.2f}({dep_:5.2f})")
    print("=" * 140)
    if depth and any(depth.values()):
        print(f"表 2：深度误差（最终模型、裁剪后 Balanced，全图，{len(names)} 对）")
        print(f"{'range':<18}{'像素数':>10}{'MAE (m)':>10}{'RMSE (m)':>10}{'AbsRel':>9}{'Disp EPE':>10}")
        for k, lst in depth.items():
            if not lst:
                continue
            de = np.concatenate([x[0] for x in lst]); gd = np.concatenate([x[1] for x in lst]); dd = np.concatenate([x[2] for x in lst])
            print(f"{k:<18}{de.size:>10d}{np.abs(de).mean():>10.3f}{math.sqrt((de ** 2).mean()):>10.3f}{(np.abs(de) / gd).mean():>9.4f}{np.abs(dd).mean():>10.3f}")
    else:
        print("表 2：没有找到 training/calib_cam_to_cam/ 标定文件，深度表跳过（需要 KITTI 2015 的 calib 压缩包）")
    print("=" * 140)
    print("把以上输出整体复制给我即可。")


if __name__ == "__main__":
    main()
