"""L-soft-B（同机 Q_evap/PLR 容量守恒）的构造正确性。

测的是约束本身的**数学性质**，不是模型学得好不好：
  - 对容量 C_i 的取值免疫（容量必须被约掉，否则等于偷偷引入了铭牌参数）
  - 确实惩罚漂移（Q/PLR 随工况变化时应当变大）
  - 停机 / 无效 plr 不进分母
  - 缺 type_id 时**报错而不是静默跳过**
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402
from physwm.train.losses import soft_b_capacity  # noqa: E402

N_CH, N_TW, N_CP, N_DP = 7, 20, 7, 8
N = 1 + N_CH + N_TW + N_CP + N_DP
F, W, H, B = 9, 16, 4, 32
TYPE_ID = torch.tensor([0] + [1] * N_CH + [2] * N_TW + [3] * N_CP + [4] * N_DP)
K_PLR = S.CANON_FIELDS["chiller"].index("plr")
K_ON = S.CANON_FIELDS["chiller"].index("on")
CH = (TYPE_ID == 1)


def _mk(plr, q, on=None):
    """plr/q: [H][B,N_CH]。构造 preds / seq_raw / seq_avail。"""
    raw = torch.zeros(B, W + H, N, F)
    av = torch.ones(B, W + H, N, F)
    for h in range(H):
        raw[:, W + h, CH, K_PLR] = plr[h]
        raw[:, W + h, CH, K_ON] = 1.0 if on is None else on[h]
    preds = [{"q_evap": q[h], "P_plant": torch.zeros(B)} for h in range(H)]
    return preds, raw, av


def test_capacity_value_is_cancelled_out():
    """把每台机的容量整体放大 100 倍，损失必须一模一样。

    容量约不掉 = 悄悄引入了铭牌参数，而 §12 G8 明确数据里没有铭牌。
    """
    torch.manual_seed(0)
    plr = [torch.rand(B, N_CH) * 0.6 + 0.3 for _ in range(H)]
    cap = torch.rand(N_CH) * 500 + 500
    q = [p * cap[None, :] for p in plr]
    l1, _ = soft_b_capacity(*_mk(plr, q), W, TYPE_ID)
    q2 = [p * cap[None, :] * 100.0 for p in plr]
    l2, _ = soft_b_capacity(*_mk(plr, q2), W, TYPE_ID)
    assert torch.allclose(l1, l2, atol=1e-6), f"容量未被约掉：{l1} vs {l2}"


def test_perfect_proportionality_gives_zero_loss():
    torch.manual_seed(1)
    plr = [torch.rand(B, N_CH) * 0.6 + 0.3 for _ in range(H)]
    cap = torch.rand(N_CH) * 500 + 500
    q = [p * cap[None, :] for p in plr]
    loss, _ = soft_b_capacity(*_mk(plr, q), W, TYPE_ID)
    assert float(loss) < 1e-8, f"完全成比例时损失应为 0，实得 {float(loss)}"


def test_drift_is_penalised():
    """Q/PLR 随工况漂移必须让损失变大 —— 否则这个约束什么都没约束。"""
    torch.manual_seed(2)
    plr = [torch.rand(B, N_CH) * 0.6 + 0.3 for _ in range(H)]
    cap = torch.rand(N_CH) * 500 + 500
    q_ok = [p * cap[None, :] for p in plr]
    # 让「容量」随负荷率漂移（物理上不该发生）
    q_bad = [p * cap[None, :] * (1.0 + 0.8 * p) for p in plr]
    l_ok, _ = soft_b_capacity(*_mk(plr, q_ok), W, TYPE_ID)
    l_bad, _ = soft_b_capacity(*_mk(plr, q_bad), W, TYPE_ID)
    assert float(l_bad) > float(l_ok) + 1e-6, "漂移未被惩罚"


def test_off_and_invalid_plr_excluded():
    """停机机与 plr 无效的样本必须不进统计，否则违例被稀释。"""
    torch.manual_seed(3)
    plr = [torch.rand(B, N_CH) * 0.6 + 0.3 for _ in range(H)]
    cap = torch.rand(N_CH) * 500 + 500
    q = [p * cap[None, :] for p in plr]
    # 把后 3 台停机，并给它们塞入完全不成比例的 q
    on = [torch.cat([torch.ones(B, N_CH - 3), torch.zeros(B, 3)], -1) for _ in range(H)]
    q_bad = [x.clone() for x in q]
    for h in range(H):
        q_bad[h][:, -3:] = torch.rand(B, 3) * 1e4
    l, _ = soft_b_capacity(*_mk(plr, q_bad, on=on), W, TYPE_ID)
    assert float(l) < 1e-8, f"停机机被算进了损失：{float(l)}"


def test_missing_type_id_raises_not_skips():
    """静默跳过会让整个 P4 臂在「约束已开」的假象下跑完。"""
    from physwm.train.losses import LossWeights, compute_loss

    class _M:
        pass
    wts = LossWeights(lam_soft_b=1.0)
    preds = [{"P_plant": torch.zeros(B), "q_evap": torch.ones(B, N_CH)}]
    out = {"preds": preds}
    batch = {"seq_raw": torch.zeros(B, W + 1, N, F),
             "seq_avail": torch.ones(B, W + 1, N, F),
             "steady": torch.ones(B, 1),
             "P_plant": torch.zeros(B, 1), "P_fam": torch.zeros(B, 1, 4)}

    class _Obs:
        def __call__(self, *a, **k):
            return {k_: torch.zeros(()) for k_ in ("k1", "k2", "k3", "k4", "k5")}
    with pytest.raises(RuntimeError, match="_type_id"):
        compute_loss(_M(), out, batch, _Obs(), wts, W)


def test_loss_weight_defaults_match_cli():
    """`LossWeights` 的默认值必须与 train_yb3.py 的 CLI 默认一致。

    两处漂移会造成：直接构造 LossWeights() 的调用方（测试、工具、
    smoke）跑的是与正式训练**不同的**损失，而这件事不会以任何方式报错。
    实测踩过：lam_evap_bal 的 dataclass 默认是 0.0、CLI 是 0.1，
    结果 dt_evap_mode='soft' 的守卫在无关测试里炸掉。
    """
    import re
    from pathlib import Path
    from physwm.train.losses import LossWeights
    src = (Path(__file__).resolve().parents[1] / "experiments" / "train_yb3.py").read_text(encoding="utf-8")
    d = LossWeights()
    for cli, attr in (("--lam-lat", "lam_lat"), ("--lam-lip", "lam_lip"),
                      ("--lam-evap-bal", "lam_evap_bal"), ("--lam-soft-b", "lam_soft_b")):
        m = re.search(rf'"{re.escape(cli)}",\s*type=float,\s*default=([0-9.eE+-]+)', src)
        assert m, f"CLI 里找不到 {cli}"
        assert abs(float(m.group(1)) - getattr(d, attr)) < 1e-12, (
            f"{cli} 的 CLI 默认 {m.group(1)} 与 LossWeights.{attr} "
            f"默认 {getattr(d, attr)} 不一致")


def test_all_python_sources_parse():
    """全仓 .py 必须能解析。

    测试不 import `experiments/`、`tools/` 下的脚本，于是那里的语法错误
    只会在 GPU 上跑起来时才暴露 —— 实测浪费过一次调度。
    """
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    bad = []
    for d in ("physwm", "experiments", "tools", "tests"):
        for f in (root / d).rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            try:
                ast.parse(f.read_text(encoding="utf-8"))
            except SyntaxError as e:
                bad.append(f"{f.relative_to(root)}:{e.lineno} {e.msg}")
    assert not bad, "语法错误:\n" + "\n".join(bad)
