"""
把每个论文模型 checkpoint 里保存的训练配置导出成 YAML，放进仓库的 configs/paper/，保证别人能按完全相同的设置复现。
同时：根据权重里是否有 PRN / Gate 参数写入 model.use_prn / model.use_gate；把服务器上的绝对路径换成相对路径。
不需要 GPU，几秒钟。

在 stereo-matching 项目根目录运行：
    python test_onnx/SwitchStereo_PC/export_configs.py
生成 configs/paper/*.yaml，把整个 configs/paper/ 上传到 GitHub 仓库。
"""
import os
import sys

import torch
import yaml

RUNS = [  # (checkpoint 文件夹, 导出文件名, 论文中的名字)
    ("checkpoints_sceneflow_pretrain_1", "sceneflow_pretrain", "Scene Flow pre-training"),
    ("checkpoints_modelA", "model_A", "Table 3, model A / Fast engine"),
    ("checkpoints_modelrpn", "model_B", "Table 3, model B"),
    ("checkpoints_baseline", "model_C", "Table 3, model C / Table 4, L1"),
    ("checkpoints_ohem", "model_ohem", "Table 4, OHEM-L1"),
    ("checkpoints_ohem+spatial", "final", "Table 4, final model / Balanced engine"),
    ("checkpoints_ohem_ablation/rho_0p2", "ohem_ratio_0p2", "Table 5, rho = 0.2"),
    ("checkpoints_ohem_ablation/rho_0p3", "ohem_ratio_0p3", "Table 5, rho = 0.3"),
    ("checkpoints_ohem_ablation/rho_0p4", "ohem_ratio_0p4", "Table 5, rho = 0.4"),
]
PATHS = [("/root/autodl-tmp/project/stereo-matching/SceneFlowData", "datasets/SceneFlow"),
         ("/root/autodl-tmp/kitti_mix", "datasets/kitti_mix"),
         ("/root/autodl-tmp/kitti_dataset", "datasets/kitti2015"),
         ("/root/autodl-tmp/project/stereo-matching/", "./")]


def tload(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, str):
        for a, b in PATHS:
            v = v.replace(a, b)
        if v.lower() in ("true", "ture"):     # 原配置里的 "ture" 拼写在 Python 中按真值处理
            return True
        if v.lower() == "false":
            return False
    return v


def main():
    if not os.path.isdir("configs"):
        sys.exit("请在 stereo-matching 项目根目录运行")
    os.makedirs("configs/paper", exist_ok=True)
    for d, name, desc in RUNS:
        p = os.path.join(d, "model_best.pth")
        if not os.path.exists(p):
            print(f"  跳过（找不到）：{p}")
            continue
        ck = tload(p)
        cfg = ck.get("config") if isinstance(ck, dict) else None
        if not isinstance(cfg, dict):
            print(f"  跳过（没有保存配置）：{p}")
            continue
        sd = ck.get("model_state_dict", {})
        cfg = clean(cfg)
        cfg.setdefault("model", {})
        cfg["model"]["use_prn"] = any(k.replace("module.", "", 1).startswith("progressive_refiner.") for k in sd)
        cfg["model"]["use_gate"] = any(k.replace("module.", "", 1).startswith("gate_refiner.") for k in sd)
        cfg.setdefault("logging", {})
        cfg["logging"].update(checkpoint_dir=f"./runs/{name}", tensorboard_dir=f"./runs/{name}/tb",
                              log_dir=f"./runs/{name}/logs")
        out = os.path.join("configs/paper", name + ".yaml")
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"# {desc}\n# Exported from the configuration stored in {d}/model_best.pth (epoch {ck.get('epoch', '?')}).\n")
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"  {out:<36} PRN={cfg['model']['use_prn']}  Gate={cfg['model']['use_gate']}  ({desc})")
    print("完成：把 configs/paper/ 整个上传到 GitHub。")


if __name__ == "__main__":
    main()
