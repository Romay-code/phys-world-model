"""不变量回归测试。

这里每一条都对应一个**实际踩过的坑**，不是形式化的凑数测试。
它们的共同点是：出错时不报异常、指标看着还行，只在几小时训练之后才暴露。

    python -m pytest tests/ -q          （服务器上跑，本地需 KMP_DUPLICATE_LIB_OK=TRUE）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402
from physwm.data.dataset import (WindowSpec, assert_no_leak, build_site_bundle,  # noqa: E402
                                 covered_rows, enumerate_windows, fit_normalizer,
                                 split_windows)
from physwm.model.decoder import (ETA_MAX, ETA_MIN, check_hard_constraints)  # noqa: E402
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402
from physwm.train.curriculum import Curriculum, SingleStep  # noqa: E402

CSV = ROOT / "data" / "yb3_topology_complete.csv"
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="需要 yb3 数据")


@pytest.fixture(scope="module")
def bundle():
    return build_site_bundle(CSV, "yb3", WindowSpec())


# --------------------------------------------------------------------------
# 1. 防泄漏（G7）。设计文档 §4.5.4 明确要求单元测试显式验证。
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["blocked", "chronological"])
def test_no_row_overlap_between_splits(mode):
    """任意两个 split 用到的**行集合**必须不相交。

    不能只比区间端点 —— 分块模式下 train/val/test 的行是交错的，
    端点比较会漏掉中间的重叠。
    """
    spec = WindowSpec(split_mode=mode)
    b = build_site_bundle(CSV, "yb3", spec)
    sp, T = b["splits"], b["data"].T
    cov = {k: covered_rows(sp[k], spec, T) for k in ("train", "val", "test")}
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert int((cov[a] & cov[c]).sum()) == 0, f"{a} 与 {c} 行重叠"
    assert_no_leak(sp, spec, T)


def test_window_span_is_contained_in_one_block():
    """整段 [t0-W+1, t0+H] 必须落在同一块内 —— 这是不相交的充分条件。"""
    spec = WindowSpec(split_mode="blocked")
    b = build_site_bundle(CSV, "yb3", spec)
    sp = b["splits"]
    for k in ("train", "val", "test"):
        w = sp[k]
        if not len(w):
            continue
        start, end = w[:, 2], w[:, 1] + spec.H
        assert np.all(start // spec.block_len == end // spec.block_len)


def test_windows_never_cross_segment_boundary():
    """窗口不能跨 900s 断点 —— 跨了就是把两段不连续的时间当连续用。"""
    spec = WindowSpec()
    b = build_site_bundle(CSV, "yb3", spec)
    sd = b["data"]
    segs = np.array(sd.segments)
    for _, t0, hs in b["windows"][::97]:
        end = t0 + spec.H
        assert np.any((segs[:, 0] <= hs) & (end < segs[:, 1])), \
            f"窗口 [{hs},{end}] 不在任何单个连续段内"


# --------------------------------------------------------------------------
# 2. 统计量泄漏。改成分块划分后引入的隐性泄漏，不体现在样本边界上。
# --------------------------------------------------------------------------

def test_normalizer_ignores_non_train_rows(bundle):
    """把非训练行改成极端值，归一化参数必须一字不变。

    退化写法 `slice(lo, hi)` 会横跨 val/test 块，这个测试就是为了钉死它。
    """
    spec, sd = bundle["spec"], bundle["data"]
    tr = bundle["splits"]["train"]
    base = fit_normalizer(sd, tr, spec)

    trm = covered_rows(tr, spec, sd.T)
    poisoned = type(sd)(site=sd.site, sch=sd.sch, x=sd.x.copy(),
                        avail=sd.avail.copy(), extra=sd.extra, segments=sd.segments)
    poisoned.x[~trm] = 1e6                      # 只污染非训练行
    after = fit_normalizer(poisoned, tr, spec)

    assert np.allclose(base.center, after.center), "归一化 center 受到非训练行影响"
    assert np.allclose(base.scale, after.scale), "归一化 scale 受到非训练行影响"


def test_covered_rows_matches_window_spans(bundle):
    spec = bundle["spec"]
    w = bundle["splits"]["train"][:500]
    m = covered_rows(w, spec, bundle["data"].T)
    for _, t0, hs in w[::37]:
        assert m[hs] and m[t0 + spec.H]
    assert m.sum() <= (w[:, 1].max() + spec.H) - w[:, 2].min() + 1


# --------------------------------------------------------------------------
# 3. 物理硬约束。重点测**浮点边界**，不是常规输入。
# --------------------------------------------------------------------------

# yb3 训练集实测量纲。**必须用真实值** —— 用构造函数默认值会让量纲错误
# 恰好落在断言阈值内而漏检（变异测试实测过：dT_evap 的 bug 在默认 q_scale
# 下算出 12 K，卡在 15 K 阈值内；用真实 q_scale=506 则是 36 K，能抓住）。
YB3_SCALES = dict(q_scale=506.0, w_scale=952.0, dt_scale=5.0, dt_evap_scale=5.0,
                  mcp_scale=1310.0,   # = w_scale * COP_carnot(17.2) * eta_mid(0.40) / 5.0
                  mcp_cool_scale=1500.0,  # = w_scale * (1 + 17.2*0.40) / 5.0
                  approx_scale=5.0, p_scale_tower=14.1,
                  p_scale_coolpump=56.4, p_scale_coldpump=35.5)


def _mk_model(n_dev=None):
    n_dev = n_dev or {"chiller": 7, "tower": 20, "coolpump": 7, "coldpump": 8}
    m = WorldModel(ModelConfig(), n_dev, f_max=9)
    m.decoder.set_scales(**YB3_SCALES)
    return m, n_dev


def _mk_inputs(model, n_dev, B=8, extreme=0.0):
    """extreme != 0 时把各 head 的偏置推到极端，逼出 softplus/sigmoid 的饱和。"""
    if extreme:
        with torch.no_grad():
            for h in (model.decoder.head_a, model.decoder.head_d,
                      model.decoder.head_w, model.decoder.head_eta):
                h[-1].bias.fill_(extreme)
                h[-1].weight.mul_(0.0)
    d = model.cfg.d
    z = {f: torch.randn(B, n_dev.get(f, 1), d) for f in S.FAMILIES}
    z["plant"] = torch.randn(B, 1, d)
    a = torch.rand(B, n_dev["chiller"], n_dev["tower"])
    a = a / a.sum(-1, keepdim=True)
    phys = {"wet_bulb": torch.rand(B) * 30,
            "cold_out": 5 + torch.rand(B, n_dev["chiller"]) * 8,
            "on_ch": (torch.rand(B, n_dev["chiller"]) > 0.6).float()}
    for f in ("tower", "coolpump", "coldpump"):
        phys[f"on_{f}"] = (torch.rand(B, n_dev[f]) > 0.5).float()
        phys[f"freq_{f}"] = 20 + torch.rand(B, n_dev[f]) * 35
    return z, a, phys


@pytest.mark.parametrize("extreme", [0.0, -200.0, 200.0])
def test_hard_constraints_hold_at_float_extremes(extreme):
    """softplus/sigmoid 在 fp32 下会饱和到恰好 0/1，约束必须仍然成立。

    这是踩过两次的坑：eta 用裸 sigmoid 会取到恰好 0；
    cool_dt 用裸 softplus 在 x<-88 时下溢到恰好 0（违例率一度 71%）。
    """
    model, n_dev = _mk_model()
    z, a, phys = _mk_inputs(model, n_dev, extreme=extreme)
    with torch.no_grad():
        o = model.decoder(z, a, phys)
        r = check_hard_constraints(o, on=phys["on_ch"])
    for k, v in r.items():
        assert v < 1e-5, f"违例 {k}={v}（extreme={extreme}）"
    assert float(o["cool_dt"].min()) > 0.0
    # 严格开区间，且不引用 ETA_MIN/MAX —— 否则有人把区间改成 [0,1] 时本断言
    # 会跟着退化（变异测试实测过这个漏检）。卡诺下界成立的前提是 eta 严格 <1。
    assert float(o["eta"].min()) > 0.0, "eta 饱和到 0，卡诺下界失效"
    assert float(o["eta"].max()) < 1.0, "eta 饱和到 1，冷机达到卡诺极限"
    assert ETA_MIN - 1e-6 <= float(o["eta"].min())
    assert float(o["eta"].max()) <= ETA_MAX + 1e-6


def test_energy_conservation_is_identity():
    """Q_cond = Q_evap + W 必须是恒等式，误差到浮点精度。"""
    model, n_dev = _mk_model()
    z, a, phys = _mk_inputs(model, n_dev)
    with torch.no_grad():
        o = model.decoder(z, a, phys)
    rel = ((o["q_cond"] - o["q_evap"] - o["w_chiller"]).abs()
           / o["q_cond"].abs().clamp_min(1e-6)).max()
    assert float(rel) < 1e-5


def test_carnot_bound_is_structural():
    """W > Q/COP_carnot 必须由构造保证，不靠拟合。"""
    model, n_dev = _mk_model()
    z, a, phys = _mk_inputs(model, n_dev)
    with torch.no_grad():
        o = model.decoder(z, a, phys)
    assert bool((o["w_chiller"] > o["q_evap"] / o["cop_carnot"]).all())


def test_physical_quantities_in_plausible_range_at_init():
    """硬约束全 0 不代表物理合理 —— 松 eta 区间那版 COP 中位只有 0.549。

    初始化时各物理量就应落在合理范围。这条测试是那次事故的直接产物。

    **本测试覆盖不到的**：`q_mode="free"` 的不可辨识退化。两种模式下
    `cop_actual = q/w` 都恒等于 `cop_carnot * eta`，初始化时无法区分；
    退化只在训练之后显现。那一项由 `train/loop.py::evaluate` 的 `phys`
    回读负责，属于训练期检查，单测做不到。
    """
    torch.manual_seed(0)
    model, n_dev = _mk_model()
    z, a, phys = _mk_inputs(model, n_dev)
    with torch.no_grad():
        o = model.decoder(z, a, phys)
    m = phys["on_ch"] > 0.5
    cop = float(o["cop_actual"][m].median())
    assert 1.5 < cop < 20.0, f"COP 中位 {cop:.3f} 不在物理范围（pb1实测 8.11）"

    # 蒸发侧温差。曾把它写成 `q_evap / q_scale * dt_scale`，换 from_w 参数化后
    # q_evap 量级变了 10 倍，算出 29 K（真值 5 K），k4 静默崩坏。
    dte = float((o["cold_back"] - phys["cold_out"])[m].median())
    assert 1.0 < dte < 15.0, f"蒸发侧温差 {dte:.2f} K 不在物理范围（yb3 实测 5.0）"

    # 冷却侧温差同理，yb3 实测中位 4.7 K
    cdt = float(o["cool_dt"][m].median())
    assert 0.5 < cdt < 15.0, f"冷却侧温差 {cdt:.2f} K 不在物理范围（yb3 实测 4.7）"


# --------------------------------------------------------------------------
# 4. 置换等变。加任何按索引的位置编码都会破坏它。
# --------------------------------------------------------------------------

def test_encoder_is_permutation_equivariant(bundle):
    torch.manual_seed(0)
    sd = bundle["data"]
    model = WorldModel(ModelConfig(), sd.sch.n_dev, sd.F).eval()
    N, W, B = sd.N, bundle["spec"].W, 3
    x = torch.randn(B, W, N, sd.F)
    av = (torch.rand(B, W, N, sd.F) > 0.2).float()
    desc = torch.randn(N, 32)
    sr = torch.randn(N, N, 5)
    sr = (sr + sr.transpose(0, 1)) / 2
    tid = torch.from_numpy(sd.sch.type_id)
    ctx = torch.zeros(B, model.cfg.d_ctx)

    ci = (tid == S.FAMILIES.index("chiller")).nonzero().squeeze(-1)
    perm = torch.arange(N)
    perm[ci[0]], perm[ci[1]] = ci[1].clone(), ci[0].clone()

    with torch.no_grad():
        z1, _ = model.encode(x, av, desc=desc, stat_rel=sr, type_id=tid, site_ctx=ctx)
        z2, _ = model.encode(x[:, :, perm], av[:, :, perm], desc=desc[perm],
                             stat_rel=sr[perm][:, perm], type_id=tid[perm], site_ctx=ctx)
    rel = float((z2 - z1[:, perm]).abs().max() / z1.abs().max())
    assert rel < 1e-4, f"置换等变被破坏，相对误差 {rel:.2e}"


# --------------------------------------------------------------------------
# 5. 重锚定的顺序。写反了转移网络会拿不到梯度，损失静默退化成自编码。
# --------------------------------------------------------------------------

def test_transition_gets_gradient_at_full_reanchor(bundle):
    """p=1 时转移网络**仍须**在梯度路径上。

    p=1 的语义是「每步从真值起跳推一步」≡ 单步预测，不是「跳过转移」。
    若先重锚再解码，解码的就是纯真值编码，转移网络梯度为 0。
    """
    torch.manual_seed(0)
    sd, spec = bundle["data"], bundle["spec"]
    model = WorldModel(ModelConfig(), sd.sch.n_dev, sd.F)
    B, N = 2, sd.N
    batch = {
        "seq_x": torch.randn(B, spec.W + 4, N, sd.F),
        "seq_avail": torch.ones(B, spec.W + 4, N, sd.F),
        "seq_raw": torch.rand(B, spec.W + 4, N, sd.F) * 10 + 5,
    }
    tid = torch.from_numpy(sd.sch.type_id)
    out = model.rollout(batch, desc=torch.randn(N, 32),
                        stat_rel=torch.randn(N, N, 5), type_id=tid,
                        site_ctx=torch.zeros(B, model.cfg.d_ctx),
                        H=3, W=spec.W, reanchor_p=1.0)
    sum(p["P_plant"].sum() for p in out["preds"]).backward()
    g = [p.grad.abs().sum().item() for p in model.transition.parameters()
         if p.grad is not None]
    assert g and max(g) > 0, "p=1 时转移网络没有梯度 —— 重锚顺序写反了"


# --------------------------------------------------------------------------
# 6. 课程表
# --------------------------------------------------------------------------

def test_curriculum_schedule():
    c = Curriculum(total_epochs=1000, H_max=48)
    assert c.H(0) == 1 and c.reanchor_p(0) == 1.0
    assert c.stage(0) == "t1" and c.stage(500) == "t2" and c.stage(999) == "t3"
    assert c.H(999) == 48
    assert c.reanchor_p(999) == 0.0, "t3 必须强制断真值"
    hs = [c.H(e) for e in range(1000)]
    ps = [c.reanchor_p(e) for e in range(1000)]
    assert all(b >= a for a, b in zip(hs, hs[1:])), "H 必须单调不减"
    assert all(b <= a + 1e-9 for a, b in zip(ps, ps[1:])), "p 必须单调不增"


def test_single_step_curriculum():
    c = SingleStep(total_epochs=100, H_max=48)
    assert all(c.H(e) == 1 for e in (0, 50, 99))
    assert all(c.stage(e) == "t1" for e in (0, 50, 99))


# --------------------------------------------------------------------------
# 7. avail 语义：停机的温度是「无读数」，不是 0 度
# --------------------------------------------------------------------------

def test_off_chiller_temperatures_are_masked(bundle):
    sd = bundle["data"]
    K = S.CANON_FIELDS["chiller"]
    ch = [i for i, (f, _) in enumerate(sd.sch.token_index) if f == "chiller"]
    off = sd.x[:, ch, K.index("on")] < 0.5
    for fld in S.OFF_INVALID_FIELDS:
        if fld not in K:
            continue
        a = sd.avail[:, ch, K.index(fld)]
        assert float(a[off].max()) == 0.0, f"停机处 {fld} 未被掩码"


def test_phantom_device_excluded_from_labels(bundle):
    """从未开机的设备视为不存在，不该被计成「缺标签」。"""
    e = bundle["data"].extra
    assert e["power_coldpump_n_phantom"] == 1          # yb3 的 coldpump/7
    assert bundle["info"]["power_src"]["coldpump"] == "per_device"
    assert bundle["info"]["power_src"]["tower"] == "aggregate"   # 仅 4/20 有标签
