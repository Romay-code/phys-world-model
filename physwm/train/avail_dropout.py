"""输入可用性随机丢弃 —— 让模型不敢依赖任何一条「不是每个站都有」的通道。

## 为什么需要

设计文档 §2.2 的**能力矩阵**一直只管**输出**：哪个站能监督哪些目标。
**输入侧从来没有对应的纪律** —— 没有任何机制阻止模型把某条通道当成主要证据，
而那条通道在别的站根本不存在。P5-A 的零样本失败正是这个洞：

    pc3 零样本 R² = −1.64，而 hx = 0.9335（同为留出站）

`tools/diag_avail_matrix.py` 摊开全通道后，pc3 相对 12 训练站中位缺五条，
且**全部是负荷/功率类的软测量**：

    chiller.consumption   0.000 vs 1.000     逐台冷机功率
    chiller.plr           0.000 vs 0.983     部分负荷率
    plant.tower_out       0.000 vs 1.000     冷却塔出水温（14 站里只有 pc3 没有）
    coolpump.consumption  0.000 vs 0.979
    coldpump.consumption  0.000 vs 0.940

`tools/diag_avail_ablate.py` 在熟悉的站上按同样的组合掩掉，复现了 pc3 的**签名**：

    yb3   原样 R² 0.994 -> 全掩 −0.387，corr 0.997 -> 0.047，
          预测 std/真值 std 从 0.97 掉到 **0.23**
    pc3   实测 R² −1.64，corr 0.883，预测 std/真值 std = **0.26**

即：**没有负荷类通道时，模型推不出「量级」，输出塌成近似常数**，
但仍能靠温度/频率/开机位跟住动态。这解释了 `tools/diag_zeroshot_calib.py`
测到的全部现象 —— corr 0.883 却 R² 为负，一个比例因子 k=2.303 就能把 R²
拉回 0.699。

## 做法

训练时按 batch（一个 batch 只来自一个站）随机把整条通道掩成「缺测」：
`x`、`raw` 置 0 且 `avail` 置 0 —— 与真实缺测在张量上逐位一致
（编码器把 `[x⊙avail ; avail]` 一起投影，能区分「缺测」与「值为 0」）。

两级采样，缺一不可：

  · `p_joint` 概率下**整组一起掩** —— 模拟 pc3 这种「全聚合站」。
    若只做独立采样，五条同时缺的概率是 p⁵，几乎永远不会被采到，
    而那恰恰是真实存在的一个站。
  · 否则逐条独立以 `p_each` 掩 —— 覆盖 zx / pb1 / pc2 那种部分缺失。

## 边界

**只掩不进物理解码器的通道。** `WorldModel.physical_inputs` 从 raw 里取
`wet_bulb / load / cold_out_temp / on / frequency`，这些掩掉会直接破坏物理
通路（卡诺 COP 拿 0 K 算），且它们在 14 站上本来就齐全。本模块的
`DROPPABLE` 与那份清单**无交集**，`test_droppable_excludes_physical_inputs`
守着这一条。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data import schema as S

# 可掩通道：跨站确实有缺、且只进编码器不进物理解码器的软测量。
# 顺序即 `tools/diag_avail_ablate.py` 的消融顺序，便于两处对照。
DROPPABLE: tuple[tuple[str, str], ...] = (
    ("chiller", "consumption"),
    ("chiller", "plr"),
    ("plant", "tower_out"),
    ("coolpump", "consumption"),
    ("coldpump", "consumption"),
    ("tower", "consumption"),
)

# 绝不可掩：`WorldModel.physical_inputs` 要从 raw 里读的那几条。
PHYSICAL_INPUTS: frozenset[tuple[str, str]] = frozenset({
    ("plant", "wet_bulb"), ("plant", "load"),
    ("chiller", "cold_out_temp"), ("chiller", "on"),
    ("tower", "on"), ("tower", "frequency"),
    ("coolpump", "on"), ("coolpump", "frequency"),
    ("coldpump", "on"), ("coldpump", "frequency"),
})


@dataclass
class AvailDropout:
    """`p_joint` 整组掩、否则逐条以 `p_each` 掩。两者皆 0 即完全关闭。"""

    p_joint: float = 0.15
    p_each: float = 0.20
    channels: tuple[tuple[str, str], ...] = DROPPABLE

    def __post_init__(self) -> None:
        bad = [c for c in self.channels if c in PHYSICAL_INPUTS]
        if bad:
            raise ValueError(
                f"这些通道要进物理解码器，掩掉会破坏物理通路，不可放进 DROPPABLE：{bad}")
        for n, v in (("p_joint", self.p_joint), ("p_each", self.p_each)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{n} 必须落在 [0,1]，收到 {v}")

    @property
    def enabled(self) -> bool:
        return self.p_joint > 0.0 or self.p_each > 0.0

    def pick(self, gen: torch.Generator | None = None) -> list[tuple[str, str]]:
        """抽这一个 batch 要掩掉的通道。"""
        if not self.enabled:
            return []
        r = torch.rand((), generator=gen).item()
        if r < self.p_joint:
            return list(self.channels)
        keep = torch.rand(len(self.channels), generator=gen) < self.p_each
        return [c for c, k in zip(self.channels, keep.tolist()) if k]

    def apply(self, batch: dict, sch, gen: torch.Generator | None = None
              ) -> tuple[dict, list[tuple[str, str]]]:
        """就地返回掩过的 batch 副本与实际掩掉的通道清单。

        三个张量一起改：`seq_x` 进编码器、`seq_raw` 进物理解码器、
        `seq_avail` 是掩码本身。只改一个会让两条通路看到不同的世界。
        """
        picked = self.pick(gen)
        if not picked:
            return batch, []
        out = dict(batch)
        for k in ("seq_x", "seq_avail", "seq_raw"):
            out[k] = batch[k].clone()
        hit: list[tuple[str, str]] = []
        for fam, fld in picked:
            rows = [i for i, (f, _) in enumerate(sch.token_index) if f == fam]
            if not rows or fld not in S.CANON_FIELDS[fam]:
                continue
            k = S.CANON_FIELDS[fam].index(fld)
            for key in ("seq_x", "seq_avail", "seq_raw"):
                out[key][:, :, rows, k] = 0.0
            hit.append((fam, fld))
        return out, hit
