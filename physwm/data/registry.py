"""站点登记表：哪些文件属于同一个冷站、每个站属于哪个能力档。

P5-A 的零样本失败暴露出，「留出哪些站」这件事此前只存在于实验脚本的一个
字符串参数里，没有任何结构约束。两条实测事实说明这不够：

## 一、14 个文件只对应 11 个独立冷站

`tools/diag_site_independence.py` 实测（时间轴逐位一致 + 湿球相关 > 0.99）：

    pa2_中温 == pa2_低温     湿球相关 0.9912、中位差 0.43 K
    yb_中温     == yb_低温          湿球相关 1.0000、中位差 **0.000 K**（同一个传感器）
    yc_中温     == yc_低温          湿球相关 0.9997、中位差 0.24 K

三对各是同一冷站的中温/低温两个温区回路（台数不同，是两套设备）。
**留出一个回路而把兄弟回路留在训练里，不是跨站零样本** —— 同楼、同天气、
同负荷排程。`expand_holdout` 强制把兄弟回路一起留出。

`hx` 没有兄弟回路，故 P5-A 的 hx 零样本 R²=0.9335 不受影响。

## 二、能力分档，且 D 档只有一个成员

按逐台功率标签的齐备程度分档（由 `power_src` 实测导出，不硬编码）：

    A  四族全逐台            bh, tx                                    2 站
    B  冷机+泵逐台、塔聚合     hx, yb3, pa2×2, yb×2, yc×2            5 冷站
    C  仅冷机逐台            zx, pb1, pc2          3 站
    D  全聚合                pc3                           **1 站**

**D 档 n=1 是 P5-A 失败的结构性原因**：留出 pc3 等于训练集里没有任何一个站
有那个能力模式，考的不是「零样本泛化」而是「外推到一个空档」。
`holdout_kind` 把这件事变成显式判定，报数时必须一起写。
"""
from __future__ import annotations

# 同名重复文件（同一冷站的不同导出），整个项目只在这里列一次
DUP_FILES: frozenset[str] = frozenset({
    "yb3_topology_clean_v2", "yb3_topology_complete", "yb3test", "zxtest"})

# 文件 -> 冷站。未列出的文件即自成一个冷站。
# 由 `tools/diag_site_independence.py` 实测导出，
# `tests/test_registry.py::test_plant_grouping_matches_data` 每次从数据重算核对。
PLANT_OF: dict[str, str] = {
    "pa2_中温": "pa2", "pa2_低温": "pa2",
    "yb_中温": "yb", "yb_低温": "yb",
    "yc_中温": "yc", "yc_低温": "yc",
}

TIER_DESC: dict[str, str] = {
    "A": "四族全逐台",
    "B": "冷机+泵逐台、塔聚合",
    "C": "仅冷机逐台",
    "D": "全聚合（无任何逐台功率标签）",
}


def plant_of(site: str) -> str:
    """站点文件名 -> 冷站标识。没有兄弟回路的站，冷站标识就是它自己。"""
    return PLANT_OF.get(site, site)


def siblings(site: str) -> list[str]:
    """同一冷站的全部回路（含自己）。"""
    p = plant_of(site)
    return sorted({s for s in PLANT_OF if PLANT_OF[s] == p} | {site})


def capability_tier(power_src: dict[str, str]) -> str:
    """由 `build_site_bundle` 的 `power_src` 判档。

    分档依据是**逐台功率标签**，因为它同时决定两件事：能不能做逐台监督
    （损失 k=5），以及编码器能不能拿到逐台功率这条输入通道 —— 后者正是
    P5-A 里 pc3 崩掉的那条（§13 #47）。
    """
    per_dev = {f for f, v in power_src.items() if v == "per_device"}
    if "chiller" not in per_dev:
        return "D"
    if len(per_dev) == 4:
        return "A"
    if {"coolpump", "coldpump"} <= per_dev:
        return "B"
    return "C"


def expand_holdout(names, all_sites=None) -> tuple[str, ...]:
    """把留出清单扩成**冷站完整**的 —— 点了一个回路就连兄弟回路一起留出。

    这不是便利功能，是纪律：漏掉兄弟回路会让「零样本」偏乐观，
    而且不会有任何报错。
    """
    want = {n for n in names if n}
    out = set()
    for n in want:
        out |= set(siblings(n))
    if all_sites is not None:
        out &= set(all_sites)
    return tuple(sorted(out))


def holdout_kind(held: str, train_tiers) -> str:
    """这次留出到底在考什么 —— 报数时必须与 R² 一起写。

    `train_tiers` 是训练集里出现过的能力档集合。

      同档跨站   该站的档在训练集里还有别的成员。考的是**跨站泛化**，
                 这是门限⑥「留出站点零样本 R² ≥ 0.80」该报的数。
      跨档外推   该站的档在训练集里一个成员都没有。考的是**能力外推**，
                 难度完全不同，不可与上者混报（P5-A 的 pc3 就是这一类）。
    """
    return "同档跨站" if held in train_tiers else "跨档外推"
