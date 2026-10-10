"""
一致性检查·第 1 步（在电脑上运行，CPU 即可）：
准备 Jetson 引擎要用的真实输入，并保存 PyTorch（FP32）在同样输入上的输出，供第 3 步对比。

在项目根目录（有 models/ 文件夹）运行：
    python make_consistency_inputs.py --ckpt 权重路径/model_best.pth --kitti_root 你的kitti文件夹 --use_all

生成文件夹 consistency_inputs/（约 50 MB），整个拷到 Jetson 的 ~/Desktop/test_floder/ 下。
预处理与 eval_modes_shared_weights.py 的 (A) 完全一致：原图中心 320x640，不缩放，ImageNet 归一化。
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

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

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
H, W = 320, 640
MODES = {
    "fast": dict(use_prn=False, use_gate=False),
    "balanced": dict(use_prn=True, use_gate=False),
    "accurate": dict(use_prn=True, use_gate=True),
}


def list_val(root, use_all):
    names = sorted(n for n in os.listdir(os.path.join(root, "training", "image_2")) if n.endswith("_10.png"))
    return names if use_all else names[int(len(names) * 0.8):]


def load_crop(root, sub, name, is_disp=False):
    a = Image.open(os.path.join(root, "training", sub, name))
    a = np.asarray(a, np.float32) / 256.0 if is_disp else np.asarray(a.convert("RGB"), np.float32) / 255.0
    h, w = a.shape[:2]
    y, x = (h - H) // 2, (w - W) // 2
    return a[y:y + H, x:x + W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--use_all", action="store_true", help="目录里只放了验证集 40 张时加上")
    ap.add_argument("--n", type=int, default=10, help="用几张图（默认 10）")
    ap.add_argument("--out", default="consistency_inputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = list_val(args.kitti_root, args.use_all)
    step = max(1, len(names) // args.n)
    names = names[::step][:args.n]
    stems = [n.replace(".png", "") for n in names]
    print(f"选用 {len(stems)} 张：{stems}")

    inputs = {}
    for name, stem in zip(names, stems):
        l = (load_crop(args.kitti_root, "image_2", name) - MEAN) / STD
        r = (load_crop(args.kitti_root, "image_3", name) - MEAN) / STD
        gt = load_crop(args.kitti_root, "disp_occ_0", name, is_disp=True)
        lt = np.ascontiguousarray(l.transpose(2, 0, 1)[None], dtype=np.float32)
        rt = np.ascontiguousarray(r.transpose(2, 0, 1)[None], dtype=np.float32)
        lt.tofile(os.path.join(args.out, f"left_{stem}.bin"))   # trtexec --loadInputs 读取的原始 float32
        rt.tofile(os.path.join(args.out, f"right_{stem}.bin"))
        np.save(os.path.join(args.out, f"gt_{stem}.npy"), gt)
        inputs[stem] = (torch.from_numpy(lt), torch.from_numpy(rt))

    ckpt = torch_load_cpu(args.ckpt)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    # 新的部署方案（与 make_onnx_variants.py 生成的 ONNX 一一对应）：
    #   fast         = 匹配 + PRN 在 1/8 的第一级残差修正 + 双线性放大
    #   balanced     = 裁剪后的 Balanced（p1、full_res 两级不加残差）
    #   balancedfull = 未裁剪的 Balanced（用于和原引擎对比）
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = EnhancedStereoNet(backbone_type="resnet18", max_disp=192, use_prn=True, use_gate=False)
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  !! 缺失权重 {len(missing)} 个")
    model = model.to(dev).eval()
    skip = set()
    for name, st in model.progressive_refiner.stages.items():
        st.register_forward_hook(lambda m, i, o, name=name: i[0] if name in skip else None)
    with torch.no_grad():
        for stem, (lt, rt) in inputs.items():
            lt_, rt_ = lt.to(dev), rt.to(dev)
            skip.clear()
            out = model(lt_, rt_)
            p3 = out["all_disparities"]["p3"]            # PRN 第一级（1/8）修正后的视差
            fast = torch.nn.functional.interpolate(p3, size=lt_.shape[-2:], mode="bilinear", align_corners=True) * (lt_.shape[-1] / p3.shape[-1])
            full = out["disp_init"]
            skip.update({"p1", "full_res"})
            pruned = model(lt_, rt_)["disp_init"]
            skip.clear()
            for mode, d in (("fast", fast), ("balanced", pruned), ("balancedfull", full)):
                np.save(os.path.join(args.out, f"pt_{mode}_{stem}.npy"), d.squeeze().float().cpu().numpy())
    print("  PyTorch 参考输出（fast / balanced / balancedfull）已保存")
    del model

    with open(os.path.join(args.out, "names.txt"), "w", newline="\n") as f:  # Windows 上也写 Linux 换行
        f.write("\n".join(stems) + "\n")
    print(f"完成：把整个 {args.out}/ 文件夹拷到 Jetson 的 ~/Desktop/test_floder/ 下。")


if __name__ == "__main__":
    main()
