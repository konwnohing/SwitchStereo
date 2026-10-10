"""
修正部署 ONNX 里 grid_sampler 采样节点的属性，使其与训练时（PyTorch）完全一致：
  RCM 相关性查表（rcm_matcher）       ：align_corners=1，padding_mode=0（zeros）
  PRN 特征 warping（progressive_refiner）：align_corners=1，padding_mode=1（border）
只改这两个属性，图里其他任何东西（norange / trtfix 的修改）都原样保留。

用法：把本脚本和下面 3 个 ONNX 放在同一个文件夹，运行
    python fix_onnx_grid_attrs.py
生成 3 个新文件（原文件不动）：
    esnet_fast_plugin_acfix.onnx
    esnet_balanced_plugin_norange_trtfix_acfix.onnx
    esnet_accurate_plugin_norange_trtfix_acfix.onnx
"""
import os
import sys

import onnx
from onnx import helper

TARGETS = [
    ("fast", "esnet_fast_plugin.onnx"),
    ("balanced", "esnet_balanced_plugin_norange_trtfix.onnx"),
    ("accurate", "esnet_accurate_plugin_norange_trtfix.onnx"),
]
# 每个模式应有的 (RCM 节点数, PRN 节点数)：RCM = 6 次迭代 x 4 层金字塔；PRN = p3/p2/p1/full_res 四级 warp
EXPECT = {"fast": (24, 0), "balanced": (24, 4), "accurate": (24, 4)}
SETTING = {"rcm": dict(align_corners=1, padding_mode=0), "prn": dict(align_corners=1, padding_mode=1)}


def set_int_attr(node, name, value):
    new = helper.make_attribute(name, int(value))
    for a in node.attribute:
        if a.name == name:
            a.CopyFrom(new)
            return
    node.attribute.append(new)


def classify(node):
    text = " ".join([node.name] + list(node.input) + list(node.output))
    if "rcm_matcher" in text:
        return "rcm"
    if "progressive_refiner" in text:
        return "prn"
    return None


def attrs_of(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def main():
    ok_all = True
    for mode, fname in TARGETS:
        print("=" * 70)
        if not os.path.exists(fname):
            print(f"[{mode}] 找不到 {fname}，跳过")
            ok_all = False
            continue
        model = onnx.load(fname)
        nodes = [n for n in model.graph.node if n.op_type == "grid_sampler"]
        groups = {"rcm": [], "prn": [], None: []}
        for n in nodes:
            groups[classify(n)].append(n)
        n_rcm, n_prn, n_unk = len(groups["rcm"]), len(groups["prn"]), len(groups[None])
        print(f"[{mode}] {fname}：grid_sampler 共 {len(nodes)} 个（RCM {n_rcm}，PRN {n_prn}，无法识别 {n_unk}）")
        if n_unk or (n_rcm, n_prn) != EXPECT[mode]:
            print(f"  !! 数量与预期 {EXPECT[mode]} 不一致或有无法识别的节点，未写出文件。请把这段输出发给我。")
            for n in groups[None][:5]:
                print("     无法识别：", n.name, list(n.input))
            ok_all = False
            continue
        print(f"  修改前示例：{attrs_of(nodes[0])}")
        for kind in ("rcm", "prn"):
            for n in groups[kind]:
                for k, v in SETTING[kind].items():
                    set_int_attr(n, k, v)
        for kind in ("rcm", "prn"):
            if groups[kind]:
                print(f"  修改后 {kind.upper()}：{attrs_of(groups[kind][0])}")
        out = fname.replace(".onnx", "_acfix.onnx")
        onnx.save(model, out)
        print(f"  已保存：{out}")
    print("=" * 70)
    if ok_all:
        print("全部完成。把 3 个 *_acfix.onnx 拷到 Jetson 的 ~/Desktop/test_floder/。")
    else:
        print("有文件没有处理成功，请把上面的输出发给我。")
        sys.exit(1)


if __name__ == "__main__":
    main()
