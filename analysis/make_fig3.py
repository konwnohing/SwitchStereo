"""
重新生成论文图 3（中间结果可视化），用最终模型、验证集场景，带色标、高分辨率。CPU 即可（无卡模式约 3–6 分钟），有卡约 1 分钟。

每个场景两行三列：
  第 1 行：左图 | 1/8 分辨率视差（经 PRN 第一级修正，双线性放大显示）| 最终视差（PRN 输出 D_init，即 Balanced 引擎输出）
  第 2 行：Gate 置信度 | 1/8 视差的误差 | 最终视差的误差
  （加 --show_raw 会多一列：匹配模块的原始输出 D_p3 及其误差。注意它在带 PRN 训练的模型里没有被直接监督，误差很大，见论文表 8。）
底部三条色标：视差（px）、误差（px）、置信度。误差只在有真值的像素上计算，为便于观察做了 3×3 膨胀。
Gate 开/关的输出差异会打印出来（应约 0.001 px），所以图中不再单独放“Gate 后”一列。

在 SwitchStereo_PC 目录下运行：
    P=/root/autodl-tmp/project/stereo-matching
    python make_fig3.py --ckpt $P/checkpoints_ohem+spatial/model_best.pth --kitti_root /root/autodl-tmp/kitti_dataset
输出（fig3/ 文件夹）：
    fig3.pdf / fig3.png（600 dpi）    论文用的成图
    panels/                          每个小图的全分辨率 PNG（想自己排版时用）
    candidates.jpg                   验证集 79 个场景的缩略图和编号；想换场景就用 --scenes 指定两个编号重跑
    stats.txt                        各输出的 EPE / D1，以及 Gate 开关差异
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.getcwd())
import torchvision.models as _tvm  # noqa: E402
_orig = _tvm.resnet18
_tvm.resnet18 = lambda *a, **k: _orig(weights=None)
from models.enhanced_stereonet import EnhancedStereoNet  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ERR_MAX = 5.0


def tload(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def pad32(a):
    h, w = a.shape[:2]
    tp, rp = (-h) % 32, (-w) % 32
    return np.pad(a, [(tp, 0), (0, rp)] + ([(0, 0)] if a.ndim == 3 else [])), tp


def metrics(pred, gt):
    v = (gt > 0) & (gt < 192)
    e = np.abs(pred[v] - gt[v])
    return float(e.mean()), float(((e > 3) & (e > 0.05 * gt[v])).mean() * 100)


def err_map(pred, gt):
    v = (gt > 0) & (gt < 192)
    e = np.where(v, np.abs(pred - gt), 0.0).astype(np.float32)
    vv = v.astype(np.float32)
    try:
        import cv2
        k = np.ones((3, 3), np.uint8)
        e = cv2.dilate(e, k); vv = cv2.dilate(vv, k)
    except ImportError:
        pass
    return np.ma.masked_where(vv < 0.5, np.minimum(e, ERR_MAX))


def up(d_low, shape_pad, tp, h, w):
    t = F.interpolate(d_low, size=shape_pad, mode="bilinear", align_corners=True) * (shape_pad[1] / d_low.shape[-1])
    return t.squeeze().float().cpu().numpy()[tp:tp + h, :w]


def candidates(root, names, path):
    tw, th, cols = 310, 94, 4
    rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 8) + 8, rows * (th + 22) + 8), "white")
    d = ImageDraw.Draw(sheet)
    for k, n in enumerate(names):
        im = Image.open(os.path.join(root, "image_2", n)).convert("RGB").resize((tw, th))
        x, y = 8 + (k % cols) * (tw + 8), 8 + (k // cols) * (th + 22)
        sheet.paste(im, (x, y + 16)); d.text((x, y + 2), n[:6], fill="black")
    sheet.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--scenes", nargs="+", default=["000126", "000184"], help="验证集编号（000121–000199）")
    ap.add_argument("--out", default="fig3")
    ap.add_argument("--show_raw", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    root = os.path.join(args.kitti_root, "training")
    os.makedirs(os.path.join(args.out, "panels"), exist_ok=True)
    held = sorted(n for n in os.listdir(os.path.join(root, "image_2")) if n.endswith("_10.png") and int(n[:6]) >= 121)
    candidates(root, held, os.path.join(args.out, "candidates.jpg"))
    for s in args.scenes:
        if int(s) < 121:
            sys.exit(f"!! {s} 不在验证集（000121–000199）里，训练时见过，请换一个编号")

    ck = tload(args.ckpt)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    sd = {k.replace("module.", "", 1): v for k, v in sd.items() if torch.is_tensor(v)}
    m = EnhancedStereoNet(backbone_type="resnet18", max_disp=192, use_prn=True, use_gate=True)
    miss, _ = m.load_state_dict(sd, strict=False)
    print(f"模型：{args.ckpt}（缺失权重 {len(miss)} 个，应为 0），设备 {dev}")
    m = m.to(dev).eval()
    rcm_out = {}
    m.rcm_matcher.register_forward_hook(lambda mod, i, o: rcm_out.__setitem__("o", o))

    scenes, lines = [], []
    for s in args.scenes:
        n = f"{s}_10.png"
        l = np.asarray(Image.open(os.path.join(root, "image_2", n)).convert("RGB"), np.float32) / 255.0
        r = np.asarray(Image.open(os.path.join(root, "image_3", n)).convert("RGB"), np.float32) / 255.0
        gt = np.asarray(Image.open(os.path.join(root, "disp_occ_0", n)), np.float32) / 256.0
        h, w = gt.shape
        (lp, tp), (rp, _) = pad32(l), pad32(r)
        to_t = lambda x: torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(dev)  # noqa: E731
        with torch.no_grad():
            out = m(to_t(lp), to_t(rp))
        shp = lp.shape[:2]
        res = {
            "raw": up(rcm_out["o"]["disp_lowres"], shp, tp, h, w),
            "p3": up(out["all_disparities"]["p3"], shp, tp, h, w),
            "init": out["disp_init"].squeeze().float().cpu().numpy()[tp:tp + h, :w],
            "final": out["disp_final"].squeeze().float().cpu().numpy()[tp:tp + h, :w],
            "conf": out["confidence"].squeeze().float().cpu().numpy()[tp:tp + h, :w],
        }
        st = {k: metrics(res[k], gt) for k in ("raw", "p3", "init", "final")}
        gdiff = np.abs(res["final"] - res["init"])
        lines.append(f"{s}: raw D_p3 EPE {st['raw'][0]:.3f} D1 {st['raw'][1]:.2f}% | 1/8 after first PRN stage EPE {st['p3'][0]:.3f} D1 {st['p3'][1]:.2f}% | "
                     f"D_init EPE {st['init'][0]:.3f} D1 {st['init'][1]:.2f}% | with Gate EPE {st['final'][0]:.3f} D1 {st['final'][1]:.2f}% | "
                     f"|Gate on - off| mean {gdiff.mean():.4f} px, max {gdiff.max():.4f} px")
        print("  " + lines[-1])
        scenes.append((s, l, gt, res, st))

    vmax = float(max(np.percentile(sc[3]["init"], 99) for sc in scenes))
    cols = 4 if args.show_raw else 3
    fig = plt.figure(figsize=(7.0, 1.72 * len(scenes) + 0.75))
    plt.rcParams.update({"font.size": 6.5})
    gs = gridspec.GridSpec(2 * len(scenes) + 1, cols, figure=fig, height_ratios=[1] * (2 * len(scenes)) + [0.16],
                           hspace=0.30, wspace=0.04, left=0.06, right=0.975, top=0.965, bottom=0.11)
    ims = {}
    for si, (s, l, gt, res, st) in enumerate(scenes):
        row1 = [("Left image", l, None)]
        row2 = [("Confidence", res["conf"], "conf")]
        if args.show_raw:
            row1.append((f"$D_{{p3}}$ (matcher), D1 {st['raw'][1]:.1f}%", res["raw"], "disp"))
            row2.append(("Error of $D_{p3}$", err_map(res["raw"], gt), "err"))
        row1 += [(f"1/8 disparity, D1 {st['p3'][1]:.1f}%", res["p3"], "disp"),
                 (f"Final $D_{{init}}$, D1 {st['init'][1]:.1f}%", res["init"], "disp")]
        row2 += [("Error of 1/8 disparity", err_map(res["p3"], gt), "err"),
                 ("Error of final", err_map(res["init"], gt), "err")]
        letters = iter("abcdefgh")
        row1 = [(f"({next(letters)}) {t}", im_, k) for t, im_, k in row1]
        row2 = [(f"({next(letters)}) {t}", im_, k) for t, im_, k in row2]
        for ri, row in enumerate((row1, row2)):
            for ci, (title, img, kind) in enumerate(row):
                ax = fig.add_subplot(gs[2 * si + ri, ci])
                if kind is None:
                    ax.imshow(img)
                elif kind == "disp":
                    ims["disp"] = ax.imshow(img, cmap="turbo", vmin=0, vmax=vmax)
                elif kind == "conf":
                    ims["conf"] = ax.imshow(img, cmap="viridis", vmin=0, vmax=1)
                else:
                    cm = plt.get_cmap("RdYlBu_r").copy(); cm.set_bad("black")
                    ims["err"] = ax.imshow(img, cmap=cm, vmin=0, vmax=ERR_MAX)
                ax.set_title(title, pad=2)
                ax.set_xticks([]); ax.set_yticks([])
                if ci == 0:
                    ax.set_ylabel(f"Scene {si + 1}\n({s})" if ri == 0 else "", fontsize=6.5)
                name = f"scene{si + 1}_{s}_r{ri + 1}c{ci + 1}.png"
                if kind is None:
                    Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(args.out, "panels", name))
                else:
                    plt.imsave(os.path.join(args.out, "panels", name), np.ma.filled(img, np.nan) if np.ma.isMaskedArray(img) else img,
                               cmap={"disp": "turbo", "conf": "viridis", "err": "RdYlBu_r"}[kind],
                               vmin=0, vmax={"disp": vmax, "conf": 1, "err": ERR_MAX}[kind])
    cb_spec = gs[2 * len(scenes), :].subgridspec(1, 3, wspace=0.35)
    for k, (key, label) in enumerate([("disp", "Disparity (px)"), ("err", "Absolute error (px, clipped at 5)"), ("conf", "Confidence")]):
        cax = fig.add_subplot(cb_spec[0, k])
        cb = fig.colorbar(ims[key], cax=cax, orientation="horizontal")
        cb.set_label(label, labelpad=1); cb.ax.tick_params(labelsize=6, length=2, pad=1)
    fig.savefig(os.path.join(args.out, "fig3.pdf"))
    fig.savefig(os.path.join(args.out, "fig3.png"), dpi=600)
    with open(os.path.join(args.out, "stats.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"完成：{args.out}/fig3.pdf、fig3.png（600 dpi）、panels/、candidates.jpg、stats.txt")
    print("把 fig3.png 和 stats.txt 发给我；想换场景，看 candidates.jpg 选两个编号，用 --scenes 编号1 编号2 重跑。")


if __name__ == "__main__":
    main()
