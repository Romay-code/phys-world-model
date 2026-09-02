"""选模口径的不变量。

课程训练里 H 随 epoch 变化，而 val 损失是**在某个 H 上算的**，所以
「不同 epoch 的 val」只有在 H 固定时才可比。实测踩过一次：
    - t1 (H=1)  val 约 0.05
    - t2 (H=24) val 约 0.27
    - t3 (H=48) val 约 0.15
取最小值必然选中 t1 的权重，140 个 epoch 的多步训练全部作废，
最终 h*=1、误差放大 +4715%，与纯单步训练无异 —— 且不报任何错、早停也正常工作。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data.dataset import WindowSpec, build_site_bundle  # noqa: E402
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402
from physwm.train.curriculum import Curriculum, SingleStep  # noqa: E402
from physwm.train.loop import Context, TrainConfig, train  # noqa: E402
from physwm.train.losses import LossWeights  # noqa: E402

CSV = ROOT / "data" / "yb3_topology_complete.csv"
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="需要 yb3 数据")


def test_select_H_defaults_to_curriculum_max():
    """未显式指定时，选模 H 必须是课程末期的 H（= H_max），不是当前 H。"""
    cur = Curriculum(total_epochs=200, H_max=48)
    cfg = TrainConfig(epochs=200)
    H_sel = cfg.select_H or cur.H(cfg.epochs - 1)
    assert H_sel == 48
    # 课程中途的 H 差得很远 —— 正是不可比的来源
    assert cur.H(0) == 1 and cur.H(100) < 48


def test_select_H_explicit_overrides():
    cfg = TrainConfig(epochs=200, select_H=16)
    cur = Curriculum(total_epochs=200, H_max=48)
    assert (cfg.select_H or cur.H(cfg.epochs - 1)) == 16


@pytest.mark.slow
def test_history_records_constant_select_H_while_H_varies():
    """跑一个极小的课程训练，验证 history 里 H 在变而 H_sel 恒定。

    这是端到端的检查：只测 `H_sel` 的计算式不够，要确认它真的被用在了
    评估调用上、并记进了 history。
    """
    torch.manual_seed(0)
    spec = WindowSpec()
    b = build_site_bundle(CSV, "yb3", spec)
    sd, splits = b["data"], b["splits"]
    # 只取少量窗口，跑得动就行
    splits = {k: (v[:400] if k in ("train", "val") else v) for k, v in splits.items()}

    dev = torch.device("cpu")
    N = sd.N
    ctx = Context(desc=torch.zeros(N, 32), stat_rel=torch.zeros(N, N, 5),
                  type_id=torch.from_numpy(sd.sch.type_id), W=spec.W,
                  site_ctx_dim=8)
    model = WorldModel(ModelConfig(), sd.sch.n_dev, sd.F)
    cfg = TrainConfig(epochs=8, batch_size=4, steps_per_epoch=1, eval_every=1,
                      patience=1000, log_every=100, max_eval_batches=1)
    cur = Curriculum(total_epochs=8, H_max=4)
    res = train(model, sd, splits, spec, b["norm"], ctx, cfg, cur,
                LossWeights(lam_lat=0.0), dev)

    hs = [r["H"] for r in res["history"]]
    sels = [r["H_sel"] for r in res["history"]]
    assert len(set(sels)) == 1, f"选模 H 不恒定: {sels}"
    assert sels[0] == 4, f"选模 H 应为 H_max=4，实际 {sels[0]}"
    assert len(set(hs)) > 1, f"课程的 H 应当变化，实际恒为 {hs[0]}"
