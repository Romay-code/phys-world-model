"""梯度连通性检查：每个物理量输出到底有没有被数据约束。

**为什么需要这个检查。** 之前一直在验证两件事：
  1. 硬约束是否满足（`check_hard_constraints`）
  2. 物理量是否落在合理范围（`evaluate` 的 `phys` 回读）
但漏了第三件：**这个量到底有没有梯度**。

`eta` 就是这样溜过去的（§13 #24，2026-08-17 查出，08-19 修复）：`q_evap` 不进任何
损失项，`dt_evap` 由独立 head 预测而非从 `q_evap` 导出，于是 `eta -> q_evap -> (无)`，
梯度为零。它报出来的 0.40 恰好是区间中点 [0.10,0.70] 的初始化值，落在物理范围内、
跨 seed 标准差只有 0.017 —— 看起来和"学出来的"一模一样。COP = COP_carnot x eta
因此也是无信息的。

修复后 `dt_evap = q_evap / mcp` 恢复了通路。**这个脚本必须继续跑**：修复本身也可能
再被后续改动切断，而它同样不会报任何错。

一个落在合理范围内的初始化值，和一个学出来的值，在指标上无法区分。只有查梯度能分辨。

    python tools/check_gradient_flow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402

# 解码器各 head 产出什么物理量，以及它是否**应当**被数据约束。
# expect_grad=False 的项要写明理由 —— 不能默认「没梯度也行」。
HEADS = {
    "head_a":     ("approx 塔逼近度",      True,  "经软分组 -> tower_out -> k4"),
    "head_d":     ("cool_dt 冷却水温差",   True,
                   "-> cool_out -> k4（cool_dt_mode='tied' 臂下改为由 q_cond 导出）"),
    "head_delta": ("delta 每机管路微调",   True,  "-> cool_back -> k4"),
    "head_w":     ("W 冷机功率",           True,  "-> P_plant/逐台 -> k1/k2/k3"),
    "head_eta":   ("eta 卡诺效率",         False,
                   "dt_evap_mode='soft' 下不经观测损失，只经 L-soft-B 的 lam_evap_bal；"
                   "本脚本只反向观测项，故此处应为 0 —— 由 tests 里的物理项测试覆盖"),
    "head_q":     ("Q_evap（仅 free 模式）", False, "from_w 模式下不使用该 head"),
    "mcp_net":    ("m·cp（desc 驱动，每机常数）", True,
                   "mcp -> dt_evap -> cold_back -> k4（soft 臂下经 lam_evap_bal）"),
    "head_dt_ev": ("dt_evap（仅消融臂）",     False,
                   "dt_evap_mode='derived' 下由 q_evap 导出；"
                   "'free' 臂才用它 —— 那个臂就是 #24 的病灶本身"),
}


# 非 Module 的可学习物理参数也必须查 —— 它们同样会「看着像学出来的」。
PARAMS: dict = {}   # mcp 已改为 desc 驱动的 mcp_net（Module），由 HEADS 那条路覆盖


def head_modules(dec):
    out = {}
    for name in HEADS:
        m = getattr(dec, name, None)
        if m is not None:
            out[name] = m
    for fam, m in dec.head_p.items():
        out[f"head_p[{fam}]"] = m
    return out


def main() -> int:
    torch.manual_seed(0)
    n_dev = {"chiller": 7, "tower": 20, "coolpump": 7, "coldpump": 8}
    N, F, W, H, B = 1 + sum(n_dev.values()), 9, 16, 3, 4

    model = WorldModel(ModelConfig(), n_dev, F)
    type_id = torch.tensor([0] + [1] * 7 + [2] * 20 + [3] * 7 + [4] * 8)
    batch = {
        "seq_x": torch.randn(B, W + H, N, F),
        "seq_avail": torch.ones(B, W + H, N, F),
        "seq_raw": torch.rand(B, W + H, N, F) * 10 + 5,
    }
    out = model.rollout(batch, desc=torch.randn(N, 32), stat_rel=torch.randn(N, N, 5),
                        type_id=type_id, site_ctx=torch.zeros(B, 8), H=H, W=W)

    # 用与训练**同构**的损失：只对真正进损失的那些键求和。
    # 键的清单必须与 losses.py::ObsLoss 一致，否则这个检查本身就是假的。
    SUPERVISED_KEYS = ["P_plant", "w_chiller", "w_chiller_on", "cold_back",
                       "cool_back", "cool_out", "tower_out",
                       "w_tower_dev", "w_coolpump_dev", "w_coldpump_dev"]
    loss = torch.zeros(())
    used = []
    for p in out["preds"]:
        for k in SUPERVISED_KEYS:
            if k in p:
                loss = loss + p[k].square().mean()
                if k not in used:
                    used.append(k)
    loss.backward()

    print(f"用于反向的键（与 losses.py 的 ObsLoss 一致）：{used}\n")
    mods = head_modules(model.decoder)
    width = max(len(k) for k in list(mods) + list(PARAMS))
    print(f"{'head':<{width}} {'物理量':<22} {'梯度范数':>12}  判定")
    print("-" * (width + 55))

    bad = []
    items = [(n, list(m.parameters())) for n, m in mods.items()]
    for pname in PARAMS:
        prm = getattr(model.decoder, pname, None)
        if prm is not None:
            items.append((pname, [prm]))
    for name, plist in items:
        gn = sum(float(p.grad.abs().sum()) for p in plist if p.grad is not None)
        desc, expect, why = (PARAMS.get(name)
                             or HEADS.get(name)
                             or ("塔/泵比功率系数", True, "-> 分项功率 -> k2/k5"))
        alive = gn > 1e-12
        if alive == expect:
            verdict = "OK" if expect else "OK（按设计不使用）"
        elif expect and not alive:
            verdict = "**无梯度 —— 该量未被任何数据约束**"
            bad.append((name, desc, why))
        else:
            verdict = "有梯度但设计上不该有，检查是否走错分支"
        print(f"{name:<{width}} {desc:<22} {gn:>12.3e}  {verdict}")

    print()
    if bad:
        print("发现未被约束的物理量：")
        for name, desc, why in bad:
            print(f"  - {name}（{desc}）：期望路径「{why}」不通")
        print("\n这类量的报数是初始化值，不含数据信息，**不可作为物理验证的证据**。")
        return 1
    print("全部应被约束的 head 都收到了非零梯度。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
