"""
重训 Model A（无 PRN、无 Gate）用的训练入口。

为什么不用 scripts/train.py：服务器上的 scripts/train.py 已经被 OHEM 比例实验的版本替换了
（必须带 --ohem_ratio；会把 training.loss 整个替换成只有 ohem_fraction 的字典；还会把保存目录改到
checkpoints_ohem_ablation/），用它跑 baseline.yaml 这类配置会悄悄丢掉所有损失设置。
原来写“训练总结报告”的入口脚本已经找不到了，这里按原流程重建，核心部件全部用项目里原有的：
  - 同一个 TrainerFixed（多尺度损失、物理约束、负视差惩罚、平滑项、Adam、余弦学习率、梯度裁剪）
  - 同一个 EvaluatorFixed 和同一个数据划分（kitti_mix 前 80% 训练，后 20% 即 79 对验证，384x1152 中心裁剪）
  - 每轮验证一次，按验证 D1 保存 model_best.pth，每轮保存 checkpoint_last.pth
  - 结束时写 training_summary.txt（格式与原来一致）
安全措施：保存目录里已经有 model_best.pth 就拒绝开始，避免再次覆盖已有结果。

用法（在 stereo-matching 项目根目录）：
    nohup python test_onnx/SwitchStereo_PC/train_main.py --config configs/modelA.yaml \
        --resume checkpoints_sceneflow_pretrain_1/model_best.pth > modelA.log 2>&1 &
    tail -f modelA.log        # 看进度，Ctrl+C 只退出查看，不会停止训练

中断后（关机、断电）接着训练：把 --resume ... 换成 --continue_run
    nohup python test_onnx/SwitchStereo_PC/train_main.py --config configs/modelA.yaml --continue_run >> modelA.log 2>&1 &
  会从保存目录里的 checkpoint_last.pth 接着训练：恢复权重、优化器状态和余弦学习率的位置，
  并重新评估已有的 model_best.pth 找回“目前最佳”，之后与一次跑完的效果相同。
"""
import argparse
import datetime
import math
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.getcwd())
from models.enhanced_stereonet import EnhancedStereoNet  # noqa: E402
from data.dataloader import create_dataloader  # noqa: E402
from engines.trainer import TrainerFixed  # noqa: E402
from engines.evaluator import EvaluatorFixed  # noqa: E402
from utils.config import load_config  # noqa: E402

REFS = [("Model B（checkpoints_modelrpn）", "checkpoints_modelrpn/model_best.pth"),
        ("Model C（checkpoints_baseline）", "checkpoints_baseline/model_best.pth")]
IGNORE = ("logging.", "model.use_prn", "model.use_gate", "training.loss.gate_refinement.enabled")


def tload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}" if p else str(k)))
    else:
        out[p] = d
    return out


def compare_with_refs(cfg):
    """确认 Model A 与 B、C 用的是同一套训练设置（只应在 PRN/Gate 开关和保存目录上不同）。"""
    me = flat(cfg)
    for name, path in REFS:
        if not os.path.exists(path):
            print(f"  {name}：找不到 {path}，跳过对比")
            continue
        ref = tload(path).get("config")
        if not isinstance(ref, dict):
            print(f"  {name}：checkpoint 里没有训练配置，跳过对比")
            continue
        ref = flat(ref)
        diff = [(k, ref.get(k, "（无）"), me.get(k, "（无）")) for k in sorted(set(ref) | set(me))
                if not k.startswith(IGNORE) and ref.get(k) != me.get(k)]
        print(f"  与 {name} 的训练配置对比：{'除 PRN/Gate 开关和保存目录外完全一致' if not diff else f'有 {len(diff)} 处不同'}")
        for k, a, b in diff[:30]:
            print(f"    {k}:  {name.split('（')[0]} = {a}   本次 = {b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="初始化权重（与其他 KITTI 模型相同：Scene Flow 预训练）")
    ap.add_argument("--continue_run", action="store_true", help="中断后从保存目录的 checkpoint_last.pth 接着训练")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        sys.exit("!! 没有检测到 GPU（可能是无卡模式开机）。训练需要有卡开机。")

    cfg = load_config(args.config)
    m_cfg, t_cfg, l_cfg = cfg.get("model", {}), cfg.get("training", {}), cfg.get("logging", {})
    ckdir = l_cfg.get("checkpoint_dir", "./checkpoints_tmp")
    last_p, best_p = os.path.join(ckdir, "checkpoint_last.pth"), os.path.join(ckdir, "model_best.pth")
    if args.continue_run:
        if not (os.path.exists(last_p) or os.path.exists(best_p)):
            sys.exit(f"!! {ckdir} 里没有 checkpoint_last.pth / model_best.pth，无法续训。")
    else:
        if os.path.exists(best_p):
            sys.exit(f"!! {ckdir} 里已经有 model_best.pth，为避免覆盖，拒绝开始。"
                     f"如果是中断后接着训练，请把 --resume ... 换成 --continue_run。")
        if not args.resume:
            print("未指定 --resume：从 ImageNet 预训练的 ResNet-18 主干开始（仅用于 Scene Flow 预训练；KITTI 微调请用 --resume 指定 Scene Flow 模型）")
    os.makedirs(ckdir, exist_ok=True)
    if not args.continue_run:
        with open(os.path.join(ckdir, "config_used.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    use_prn, use_gate = bool(m_cfg.get("use_prn", True)), bool(m_cfg.get("use_gate", True))
    epochs = int(t_cfg.get("num_epochs", 300))
    print("=" * 80)
    print(f"配置: {args.config}   设备: {dev}   保存到: {ckdir}")
    print(f"PRN: {'开启' if use_prn else '关闭'}   Gate: {'开启' if use_gate else '关闭'}   轮数: {epochs}   "
          f"batch: {t_cfg.get('batch_size')}   学习率: {t_cfg.get('learning_rate')}")
    print(f"初始化权重: {args.resume or 'ImageNet 主干'}" if not args.continue_run else f"续训：从 {ckdir} 接着训练")
    compare_with_refs(cfg)
    print("=" * 80)

    train_loader = create_dataloader(cfg, split="train")
    val_loader = create_dataloader(cfg, split="val")
    print(f"训练 {len(train_loader.dataset)} 对 | 验证 {len(val_loader.dataset)} 对")

    model = EnhancedStereoNet(backbone_type=m_cfg.get("backbone_type", "resnet18"),
                              max_disp=m_cfg.get("max_disp", 192), use_prn=use_prn, use_gate=use_gate).to(dev)
    if not args.continue_run and args.resume:
        ck = tload(args.resume)
        sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
        sd = {k.replace("module.", "", 1): v for k, v in sd.items() if torch.is_tensor(v)}
        msg = model.load_state_dict(sd, strict=False)
        print(f"初始化权重已载入：缺失 {len(msg.missing_keys)} 个，多余 {len(msg.unexpected_keys)} 个"
              f"（关闭 PRN/Gate 时多余的应全是 progressive_refiner / gate_refiner）")
        if msg.missing_keys:
            print("  缺失示例:", msg.missing_keys[:10])
        odd = [k for k in msg.unexpected_keys if not k.startswith(("progressive_refiner.", "gate_refiner."))]
        if odd:
            print("  !! 有意外的多余权重:", odd[:10])

    trainer = TrainerFixed(model, cfg, dev)
    evaluator = EvaluatorFixed(max_disp=m_cfg.get("max_disp", 192),
                               metrics=["epe", "bad_1.0", "bad_3.0", "d1_all"], device=dev)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(l_cfg.get("tensorboard_dir", "./tensorboard_tmp"))
    except Exception:  # noqa: BLE001
        writer = None

    t0 = time.time()
    best_d1, best_epoch, best_m, last_m = float("inf"), -1, {}, {}
    start_epoch = 0
    if args.continue_run:
        def d1_of(m):
            return float(m.get("d1_all", float("inf")))
        best_ck = None
        if os.path.exists(best_p):
            try:
                best_ck = tload(best_p)
            except Exception as e:  # noqa: BLE001
                print(f"  !! model_best.pth 读取失败（{e}），只用 checkpoint_last.pth")
        try:
            last_ck = tload(last_p)
        except Exception as e:  # noqa: BLE001
            if best_ck is None:
                sys.exit(f"!! checkpoint_last.pth 和 model_best.pth 都读不了（{e}），只能重新开始训练。")
            print(f"  !! checkpoint_last.pth 读取失败（{e}，可能关机时正在写入），改从 model_best.pth 续训")
            last_ck = best_ck
        if int(last_ck["epoch"]) >= epochs and os.path.exists(os.path.join(ckdir, "training_summary.txt")):
            print(f"  已经训完全部 {epochs} 轮（training_summary.txt 已存在），无需续训。", flush=True)
            return
        # 1) 重新评估已有的最佳权重，找回“目前最佳”
        if best_ck is not None:
            model.load_state_dict(best_ck["model_state_dict"])
            best_m = dict(evaluator.evaluate(model, val_loader))
            best_d1, best_epoch = d1_of(best_m), int(best_ck["epoch"])
            print(f"  已有最佳：第 {best_epoch} 轮，D1 {best_d1:.4f}，EPE {best_m.get('epe', -1):.4f}")
        # 2) 载入最后一轮的权重和优化器状态
        model.load_state_dict(last_ck["model_state_dict"])
        if last_ck.get("optimizer_state_dict"):
            trainer.optimizer.load_state_dict(last_ck["optimizer_state_dict"])
        start_epoch = int(last_ck["epoch"])
        last_m = dict(evaluator.evaluate(model, val_loader))
        print(f"  最后一轮：第 {start_epoch} 轮，D1 {d1_of(last_m):.4f}")
        if d1_of(last_m) < best_d1:   # 关机可能发生在“刚存完最后一轮、还没存最佳”之间
            best_d1, best_epoch, best_m = d1_of(last_m), start_epoch, dict(last_m)
            trainer.save_checkpoint(start_epoch, ckdir, is_best=True)
            print("  最后一轮比已有最佳更好，已补存为 model_best.pth")
        # 3) 余弦学习率定位到第 start_epoch 轮（与一次跑完完全一致）
        sch = trainer.scheduler
        for g in trainer.optimizer.param_groups:
            base = float(g.get("initial_lr", t_cfg.get("learning_rate", 2e-5)))
            g["initial_lr"] = base
            g["lr"] = sch.eta_min + (base - sch.eta_min) * (1 + math.cos(math.pi * start_epoch / sch.T_max)) / 2
        sch.base_lrs = [g["initial_lr"] for g in trainer.optimizer.param_groups]
        sch.last_epoch = start_epoch
        sch._last_lr = [g["lr"] for g in trainer.optimizer.param_groups]
        print(f"  从第 {start_epoch + 1} 轮继续，学习率 {trainer.optimizer.param_groups[0]['lr']:.6g}"
              f"（共 {epochs} 轮，还剩 {max(0, epochs - start_epoch)} 轮）", flush=True)
    for epoch in range(start_epoch, epochs):
        ep = epoch + 1
        tl = trainer.train_epoch(train_loader, epoch, writer)
        last_m = evaluator.evaluate(model, val_loader)
        d1 = float(last_m.get("d1_all", float("inf")))
        is_best = d1 < best_d1
        if is_best:
            best_d1, best_epoch, best_m = d1, ep, dict(last_m)
        trainer.save_checkpoint(ep, ckdir, is_best=is_best)
        trainer.scheduler.step()
        lr = trainer.optimizer.param_groups[0]["lr"]
        print(f"Epoch [{ep}/{epochs}] Loss: {tl.get('total', 0):.4f} | EPE: {last_m.get('epe', -1):.4f} | "
              f"D1-all: {d1:.2f}% | 最佳 {best_d1:.4f}（第 {best_epoch} 轮）| LR: {lr:.6g} | "
              f"已用 {datetime.timedelta(seconds=int(time.time() - t0))}", flush=True)
        if writer is not None:
            writer.add_scalar("Train/loss", tl.get("total", 0), ep)
            writer.add_scalar("Val/epe", last_m.get("epe", -1), ep)
            writer.add_scalar("Val/d1_all", d1, ep)
            writer.add_scalar("Train/lr", lr, ep)
    if writer is not None:
        writer.close()

    tick = lambda b: "✅ 开启" if b else "❎ 关闭"  # noqa: E731
    lines = [
        "=" * 50, "🚀 EnhancedStereoNet 训练总结报告", "=" * 50, "",
        "🕒 [运行信息]", "-" * 50,
        f"完成时间     : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"总耗时       : {datetime.timedelta(seconds=int(time.time() - t0))}"
        + (f"（第 {start_epoch + 1} 轮起续训部分）" if start_epoch else ""),
        f"配置文件     : {args.config}",
        f"初始化权重   : {args.resume or '（续训，见首次运行的日志）'}",
        f"运行设备     : {dev}", "",
        "🏆 [最佳记录]", "-" * 50,
        f"最佳 Epoch   : 第 {best_epoch} 轮",
        f"最佳指标     : {best_d1:.4f} (基于 D1-all 判定)",
        f"最佳轮 EPE   : {best_m.get('epe', float('nan')):.4f} 像素", "",
        f"📊 [最终评估指标 (Epoch {epochs})]", "-" * 50,
        f"EPE          : {last_m.get('epe', float('nan')):.4f} 像素",
        f"D1-all       : {last_m.get('d1_all', float('nan')):.2f} %",
        f"Bad 1.0      : {last_m.get('bad_1.0', float('nan')):.2f} %",
        f"Bad 3.0      : {last_m.get('bad_3.0', float('nan')):.2f} %", "",
        "⚙️ [核心超参数 & 消融状态]", "-" * 50,
        f"主干网络     : {m_cfg.get('backbone_type', 'resnet18')}",
        f"最大视差     : {m_cfg.get('max_disp', 192)}",
        f"PRN 模块     : {tick(use_prn)}",
        f"Gate 模块    : {tick(use_gate)}",
        f"Batch Size   : {t_cfg.get('batch_size')}",
        f"学习率       : {t_cfg.get('learning_rate')}",
        f"总 Epoch 数  : {epochs}", "=" * 50,
    ]
    with open(os.path.join(ckdir, "training_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
