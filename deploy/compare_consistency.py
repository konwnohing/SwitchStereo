"""
一致性检查·第 3 步（在电脑上运行，不需要 PyTorch 模型，只用 numpy）：
对比 Jetson 引擎输出（consistency_outputs/）与 PyTorch FP32 输出（consistency_inputs/）。

    python compare_consistency.py --inputs consistency_inputs --outputs consistency_outputs

输出一张表：每个引擎与 PyTorch 的逐像素差异，以及两者各自相对真值的 EPE / D1-all。
"""
import argparse
import json
import os

import numpy as np

H, W, MAX_DISP = 320, 640, 192


def load_trt(path):
    with open(path) as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else [data]
    entry = next((e for e in entries if e.get("name") == "disparity"), entries[-1])
    return np.asarray(entry["values"], dtype=np.float32).reshape(H, W)


class Acc:
    def __init__(self):
        self.e = self.bad = self.n = 0.0

    def add(self, pred, gt):
        v = (gt > 0) & (gt < MAX_DISP)
        err = np.abs(pred[v] - gt[v])
        self.e += err.sum()
        self.bad += ((err > 3) & (err > 0.05 * gt[v])).sum()
        self.n += v.sum()

    def res(self):
        return self.e / max(self.n, 1), self.bad / max(self.n, 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="consistency_inputs")
    ap.add_argument("--outputs", default="consistency_outputs")
    args = ap.parse_args()
    stems = [s.strip() for s in open(os.path.join(args.inputs, "names.txt")) if s.strip()]

    rows = []
    for tag in sorted(os.listdir(args.outputs)):
        d = os.path.join(args.outputs, tag)
        if not os.path.isdir(d) or "_" not in tag:
            continue
        mode = tag.split("_", 1)[1]
        diffs, acc_pt, acc_trt, n_ok = [], Acc(), Acc(), 0
        for s in stems:
            p = os.path.join(d, f"{s}.json")
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                continue
            trt = load_trt(p)
            pt = np.load(os.path.join(args.inputs, f"pt_{mode}_{s}.npy")).reshape(H, W)
            gt = np.load(os.path.join(args.inputs, f"gt_{s}.npy")).reshape(H, W)
            diffs.append(np.abs(trt - pt).ravel())
            acc_pt.add(pt, gt)
            acc_trt.add(trt, gt)
            n_ok += 1
        if not n_ok:
            rows.append((tag, 0) + (float("nan"),) * 7)
            continue
        dd = np.concatenate(diffs)
        rows.append((tag, n_ok, dd.mean(), np.percentile(dd, 99), (dd > 1).mean() * 100,
                     *acc_pt.res(), *acc_trt.res()))

    print("=" * 104)
    print(f"{'engine':<14}{'images':>7} | {'mean|TRT-PT|':>12} {'99%|TRT-PT|':>12} {'>1px %':>8} |"
          f" {'PyTorch EPE':>11} {'D1%':>6} | {'TRT EPE':>8} {'D1%':>6}")
    print("-" * 104)
    for t, n, m, p99, g1, e_pt, d_pt, e_trt, d_trt in rows:
        print(f"{t:<14}{n:>7} | {m:12.3f} {p99:12.3f} {g1:8.2f} | {e_pt:11.3f} {d_pt:6.2f} | {e_trt:8.3f} {d_trt:6.2f}")
    print("=" * 104)
    print("new_* = 修正后的新引擎，old_* = 修正前的旧引擎。把这张表整体复制给我即可。")


if __name__ == "__main__":
    main()
