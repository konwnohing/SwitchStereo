"""
Gate 作用检查（有卡约 3–5 分钟）：对每个带 Gate 的模型，在 79 对验证集（384x1152 中心裁剪，与训练时验证一致）上
  1) 比较推理时 Gate 开 / 关的精度（同一套权重）；
  2) 统计 Gate 实际修正了多少：|开 - 关| 的平均值、超过 0.5 / 1 像素的像素比例、置信度分布、残差头输出大小；
  3) 打印该模型训练时 Gate 损失是否开启、权重多少。
用来区分“Gate 在训练时起作用（改善主干网络）”和“Gate 在推理时起作用（直接修正视差）”。

在 SwitchStereo_PC 目录下运行：
    P=/root/autodl-tmp/project/stereo-matching
    python check_gate.py --proj $P --kitti_root /root/autodl-tmp/kitti_dataset
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
import torchvision.models as _tvm  # noqa: E402
_orig = _tvm.resnet18
_tvm.resnet18 = lambda *a, **k: _orig(weights=None)
from models.enhanced_stereonet import EnhancedStereoNet  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
RUNS = [("C: PRN+Gate, L1", "checkpoints_baseline"),
        ("PRN+Gate, OHEM", "checkpoints_ohem"),
        ("Final: PRN+Gate, OHEM+edge", "checkpoints_ohem+spatial"),
        ("Scene Flow pretrain", "checkpoints_sceneflow_pretrain_1")]


def tload(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def crop(a, CH=384, CW=1152):
    h, w = a.shape[:2]
    a = np.pad(a, [(0, max(0, CH - h)), (0, max(0, CW - w))] + ([(0, 0)] if a.ndim == 3 else []))
    h, w = a.shape[:2]
    t, l = int(round((h - CH) / 2.0)), int(round((w - CW) / 2.0))
    return a[t:t + CH, l:l + CW]


def to_t(x, dev):
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(dev)


def metrics(pred, gt):
    v = (gt > 0) & (gt < 192)
    e = np.abs(pred[v] - gt[v])
    return e.mean(), ((e > 3) & (e > 0.05 * gt[v])).mean() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    ap.add_argument("--kitti_root", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join(args.kitti_root, "training")
    names = sorted(n for n in os.listdir(os.path.join(root, "image_2")) if n.endswith("_10.png") and int(n[:6]) >= 121)
    rd = lambda sub, n: np.asarray(Image.open(os.path.join(root, sub, n)).convert("RGB"), np.float32) / 255.0  # noqa: E731
    data = [(crop(rd("image_2", n)), crop(rd("image_3", n)),
             crop(np.asarray(Image.open(os.path.join(root, "disp_occ_0", n)), np.float32) / 256.0)) for n in names]
    print(f"验证集 {len(names)} 对（384x1152 裁剪），设备 {dev}")

    rows = []
    for label, d in RUNS:
        path = os.path.join(args.proj, d, "model_best.pth")
        if not os.path.exists(path):
            print(f"  跳过（找不到）：{path}")
            continue
        ck = tload(path)
        cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
        gcfg = (((cfg or {}).get("training", {}) or {}).get("loss", {}) or {}).get("gate_refinement", {}) or {}
        sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
        sd = {k.replace("module.", "", 1): v for k, v in sd.items() if torch.is_tensor(v)}
        m = EnhancedStereoNet(backbone_type="resnet18", max_disp=192, use_prn=True, use_gate=True)
        missing, _ = m.load_state_dict(sd, strict=False)
        n_gate_missing = sum(k.startswith("gate_refiner") for k in missing)
        if n_gate_missing:
            print(f"  跳过 {label}：权重里没有 Gate（缺 {n_gate_missing} 个）")
            continue
        m = m.to(dev).eval()
        res_out = {}
        m.gate_refiner.res_head.register_forward_hook(lambda mod, i, o: res_out.__setitem__("r", o))
        acc = {"init": [], "final": []}
        corr, conf_all, res_all = [], [], []
        with torch.no_grad():
            for l, r, gt in data:
                out = m(to_t(l, dev), to_t(r, dev))
                di = out["disp_init"].squeeze().float().cpu().numpy()
                df = out["disp_final"].squeeze().float().cpu().numpy()
                cf = out["confidence"].squeeze().float().cpu().numpy()
                rs = (res_out["r"] * m.gate_refiner.max_res).squeeze().float().cpu().numpy()
                acc["init"].append(metrics(di, gt)); acc["final"].append(metrics(df, gt))
                corr.append(np.abs(df - di).ravel()); conf_all.append(cf.ravel()); res_all.append(np.abs(rs).ravel())
        c, cf, rs = np.concatenate(corr), np.concatenate(conf_all), np.concatenate(res_all)
        ei, di_ = np.mean([a[0] for a in acc["init"]]), np.mean([a[1] for a in acc["init"]])
        ef, df_ = np.mean([a[0] for a in acc["final"]]), np.mean([a[1] for a in acc["final"]])
        rows.append((label, gcfg.get("enabled", "?"), gcfg.get("weight", "?"), ei, di_, ef, df_,
                     c.mean(), np.percentile(c, 99), (c > 0.5).mean() * 100, (c > 1).mean() * 100,
                     cf.mean(), (cf < 0.5).mean() * 100, rs.mean()))
        print(f"  完成：{label}")
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 150)
    print(f"{'model':<28}{'Gate损失':>9}{'权重':>6} | {'Gate关 EPE':>10} {'D1':>6} | {'Gate开 EPE':>10} {'D1':>6} | "
          f"{'|修正|均值':>10} {'99%':>7} {'>0.5px%':>8} {'>1px%':>7} | {'置信度均值':>10} {'<0.5占比%':>9} {'|残差头|均值':>12}")
    print("-" * 150)
    for (lb, en, w, ei, di_, ef, df_, cm, c99, c05, c1, cfm, cfl, rsm) in rows:
        print(f"{lb:<28}{str(en):>9}{str(w):>6} | {ei:10.3f} {di_:6.2f} | {ef:10.3f} {df_:6.2f} | "
              f"{cm:10.4f} {c99:7.3f} {c05:8.3f} {c1:7.3f} | {cfm:10.4f} {cfl:9.3f} {rsm:12.4f}")
    print("=" * 150)

    # 顺便打印各模型训练时实际用了哪些损失项（写论文表 3、表 4 的设置说明要用）
    print("各模型训练配置（来自 checkpoint 内保存的 config）")
    keys = [("数据", "data.root_dir"), ("裁剪H", "data.crop_height"), ("裁剪W", "data.crop_width"),
            ("轮数", "training.num_epochs"), ("OHEM比例", "training.loss.ohem_fraction"),
            ("平滑开", "training.loss.smoothness.enabled"), ("平滑权重", "training.loss.smoothness.weight"),
            ("梯度一致开", "training.loss.spatial_grad.enabled"), ("梯度一致权重", "training.loss.spatial_grad.weight"),
            ("Gate损失开", "training.loss.gate_refinement.enabled"), ("物理约束开", "training.loss.physics_constraint.enabled")]
    for d in ["checkpoints_modelA", "checkpoints_modelrpn", "checkpoints_baseline", "checkpoints_ohem",
              "checkpoints_ohem+spatial", "checkpoints_sceneflow_pretrain_1"]:
        path = os.path.join(args.proj, d, "model_best.pth")
        if not os.path.exists(path):
            continue
        ck = tload(path)
        cfg = ck.get("config") if isinstance(ck, dict) else None
        if not isinstance(cfg, dict):
            print(f"  {d}: 没有保存配置")
            continue
        vals = []
        for name, k in keys:
            v = cfg
            for part in k.split("."):
                v = v.get(part, "–") if isinstance(v, dict) else "–"
            vals.append(f"{name}={v}")
        print(f"  {d} (epoch {ck.get('epoch', '?')}): " + "  ".join(vals))
    print("=" * 150)
    print("把以上全部输出发给我。")


if __name__ == "__main__":
    main()
