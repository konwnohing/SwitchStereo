"""
从已修正采样属性的 Balanced ONNX（esnet_balanced_plugin_norange_trtfix_acfix.onnx）直接改图，生成两个新的部署模型，
原来的 norange / trtfix / 插件属性修改全部保留，不需要重新导出：

  esnet_fast_v2_acfix.onnx        新 Fast：匹配 + PRN 在 1/8 分辨率的第一级残差修正 + 双线性放大到全分辨率
                                  （对应 eval_variants.py 中的 bilinear 变体，79 对验证 D1 约 6.2%）
  esnet_balanced_pruned_acfix.onnx 裁剪后的 Balanced：去掉 PRN 中 p1、full_res 两级的残差分支
                                  （对应 pruned 变体，输出与原 Balanced 相同，D1 约 2.88%）
改图后再删除所有对输出无用的节点（全分辨率特征重建等会被整块删掉）。

用法（在放 ONNX 的文件夹里，需要 onnx 包）：
    python make_onnx_variants.py
"""
import re
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

SRC = "esnet_balanced_plugin_norange_trtfix_acfix.onnx"


def stage_add(g, lvl):
    """找到 PRN 某一级 'disp + residual' 的 Add 节点，返回 (Add 节点, 输入视差张量名)。"""
    pat = re.compile(r"/progressive_refiner/%s(_\d+)?/res_net/res_net\.6/Conv" % lvl)
    convs = [n for n in g.node if n.op_type == "Conv" and
             (pat.search(n.name) or any(("stages.%s.res_net.6.weight" % lvl) in i for i in n.input))]
    if len(convs) != 1:
        sys.exit("!! 找不到 %s 级残差输出卷积（找到 %d 个）。请把下面这些节点名发给我：\n%s" % (
            lvl, len(convs), "\n".join(n.name for n in g.node if "res_net.6" in n.name or "res_net" in n.name)[:3000]))
    r = convs[0].output[0]
    adds = [n for n in g.node if n.op_type == "Add" and r in n.input]
    if len(adds) != 1:
        sys.exit("!! %s 级残差卷积的输出没有接到唯一的 Add 上（找到 %d 个），请把这段输出发给我。" % (lvl, len(adds)))
    disp_in = [i for i in adds[0].input if i != r][0]
    return adds[0], disp_in


def dce(g):
    """只保留计算图输出所需的节点和权重。"""
    prod = {o: n for n in g.node for o in n.output}
    keep, stack = set(), [o.name for o in g.output]
    while stack:
        n = prod.get(stack.pop())
        if n is None or id(n) in keep:
            continue
        keep.add(id(n))
        stack.extend(i for i in n.input if i)
    nodes = [n for n in g.node if id(n) in keep]
    del g.node[:]; g.node.extend(nodes)
    used = {i for n in g.node for i in n.input}
    inits = [t for t in g.initializer if t.name in used]
    del g.initializer[:]; g.initializer.extend(inits)
    produced = {o for n in g.node for o in n.output}
    vis = [v for v in g.value_info if v.name in produced]
    del g.value_info[:]; g.value_info.extend(vis)


def summary(g, tag):
    gs = sum(1 for n in g.node if n.op_type == "grid_sampler")
    fr = sum(1 for n in g.node if "full_res_recon" in n.name)
    p1e = sum(1 for n in g.node if "p1_enhance" in n.name)
    print(f"  {tag}: 节点 {len(g.node)} 个，grid_sampler {gs} 个，full_res_recon 相关 {fr} 个，p1_enhance 相关 {p1e} 个，权重张量 {len(g.initializer)} 个")


def make_pruned(src, dst):
    m = onnx.load(src); g = m.graph
    summary(g, "原 Balanced")
    for lvl in ("p1", "full_res"):
        add, disp_in = stage_add(g, lvl)
        add.op_type = "Identity"           # 输出名不变（最后一级的输出就是 'disparity'），只是不再加残差
        del add.input[:]; add.input.append(disp_in)
        del add.attribute[:]
    dce(g)
    summary(g, "裁剪后 Balanced")
    onnx.save(m, dst)
    print(f"  已保存 {dst}")


def make_fast(src, dst):
    m = onnx.load(src); g = m.graph
    add, _ = stage_add(g, "p3")
    t3 = add.output[0]
    out_name = g.output[0].name
    # 原来产生最终输出的节点改个输出名，让新的分支来产生 'disparity'
    for n in g.node:
        for k, o in enumerate(n.output):
            if o == out_name:
                n.output[k] = out_name + "_unused"
    g.initializer.extend([
        numpy_helper.from_array(np.array([], dtype=np.float32), "fv2_roi"),
        numpy_helper.from_array(np.array([1, 1, 8, 8], dtype=np.float32), "fv2_scales"),
        numpy_helper.from_array(np.array(8.0, dtype=np.float32), "fv2_eight"),
    ])
    g.node.extend([
        helper.make_node("Resize", [t3, "fv2_roi", "fv2_scales"], ["fv2_up"], name="fast_v2/Resize",
                         mode="linear", coordinate_transformation_mode="align_corners"),
        helper.make_node("Mul", ["fv2_up", "fv2_eight"], [out_name], name="fast_v2/Mul"),
    ])
    dce(g)
    summary(g, "新 Fast")
    onnx.save(m, dst)
    print(f"  已保存 {dst}")


if __name__ == "__main__":
    print("=" * 80)
    make_pruned(SRC, "esnet_balanced_pruned_acfix.onnx")
    print("=" * 80)
    make_fast(SRC, "esnet_fast_v2_acfix.onnx")
    print("=" * 80)
    print("预期：裁剪后 Balanced 的 grid_sampler 为 26 个（RCM 24 + PRN 2），新 Fast 为 25 个（RCM 24 + PRN 1），")
    print("      两者 full_res_recon 相关节点都应为 0。把以上输出发给我。")
