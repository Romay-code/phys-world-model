"""三阶段课程（设计文档 §4.5.1）。

| 阶段 | epoch 占比 | H | 重锚概率 p | 含义 |
|---|---|---|---|---|
| t1 单步 | 0-30%   | 1          | 1.0        | 对齐单步能力，先把精度打上去 |
| t2 渐变 | 30-85%  | 1 -> H_max | 1.0 -> 0.0 | 逐步断开真值供给 |
| t3 想象 | 85-100% | H_max      | 强制 0.0   | 纯自产隐状态微调 |

H 与 p 一起变，是为了让 t2 早期便宜（H 小）、晚期才付满算力。若 t2 全程按
H_max 跑，计算量是不必要的 3 倍。

三条来自参考项目 v2/v3 的实测修复项，形式变了但意图不变：
warmup 期给真值、**末期强制断真值**、**选模用断真值口径**。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Curriculum:
    total_epochs: int
    H_max: int = 48
    t1_frac: float = 0.30
    t2_frac: float = 0.85     # t2 结束点（t3 从这里到 1.0）
    p_start: float = 1.0
    p_end: float = 0.0
    # mono 退火：前 20% epoch 线性升到目标值（修参考项目 v3 的 seed 崩坏）
    mono_warm_frac: float = 0.20

    def stage(self, epoch: int) -> str:
        f = epoch / max(self.total_epochs - 1, 1)
        if f < self.t1_frac:
            return "t1"
        return "t2" if f < self.t2_frac else "t3"

    def H(self, epoch: int) -> int:
        f = epoch / max(self.total_epochs - 1, 1)
        if f < self.t1_frac:
            return 1
        if f >= self.t2_frac:
            return self.H_max
        u = (f - self.t1_frac) / max(self.t2_frac - self.t1_frac, 1e-9)
        return max(1, min(self.H_max, int(round(1 + u * (self.H_max - 1)))))

    def reanchor_p(self, epoch: int) -> float:
        f = epoch / max(self.total_epochs - 1, 1)
        if f < self.t1_frac:
            return self.p_start
        if f >= self.t2_frac:
            return 0.0        # t3 强制断真值
        u = (f - self.t1_frac) / max(self.t2_frac - self.t1_frac, 1e-9)
        return self.p_start + u * (self.p_end - self.p_start)

    def mono_scale(self, epoch: int) -> float:
        f = epoch / max(self.total_epochs - 1, 1)
        return min(1.0, f / max(self.mono_warm_frac, 1e-9))

    def describe(self) -> str:
        rows = []
        for e in sorted({0, int(self.total_epochs * self.t1_frac) - 1,
                         int(self.total_epochs * self.t1_frac),
                         self.total_epochs // 2,
                         int(self.total_epochs * self.t2_frac) - 1,
                         int(self.total_epochs * self.t2_frac),
                         self.total_epochs - 1}):
            e = max(0, min(e, self.total_epochs - 1))
            rows.append(f"  epoch {e:5d}  {self.stage(e)}  H={self.H(e):3d}  "
                        f"p={self.reanchor_p(e):.3f}")
        return "\n".join(rows)


@dataclass
class SingleStep(Curriculum):
    """P2 用：只跑 t1（H=1），用于与单步基线对齐。

    H=1 时重锚无实际作用（只有一步，没有「往下传」），故 p 取 0 省掉真值编码
    那次前向 —— 它在 H=1 下纯属浪费。语义与 p=1 等价。
    """

    def stage(self, epoch: int) -> str:
        return "t1"

    def H(self, epoch: int) -> int:
        return 1

    def reanchor_p(self, epoch: int) -> float:
        return 0.0
