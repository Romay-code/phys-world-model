"""每个物理量输出必须被数据约束（有梯度）。

**这是第三类检查**，前两类都发现不了同一个问题：
    1. 硬约束违例率 —— eta 恒在 [0.10,0.70] 内，违例率 0.000
    2. 物理量合理性回读 —— eta = 0.387 ± 0.017，落在物理范围内
    3. **梯度连通性** —— eta 梯度 0.000e+00，从未被任何数据约束

一个落在合理范围内的**初始化值**，与一个学出来的值，在前两类检查里完全一致。
而且 eta 跨 5 seed 的标准差只有 0.017，比任何真正被优化的量都稳 ——
「过于稳定」本该是线索，当时被当成了好现象。

历史成因：为修 dT_evap 的量纲失配（q_evap 量级从 ~500 变 ~3700 导致算出 29K），
把 `cold_back = cold_out + q_evap/q_scale*dt_scale` 改成独立 head 预测 dt_evap，
**顺手切断了 eta 唯一的梯度通路**。修一个 bug 造出另一个。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402

N_DEV = {"chiller": 7, "tower": 20, "coolpump": 7, "coldpump": 8}
N = 1 + sum(N_DEV.values())
F, W, H, B = 9, 16, 3, 4

# 与 losses.py::ObsLoss 实际引用的键保持一致。
# 这份清单若与 losses.py 漂移，本测试就是假的 —— 见 test_supervised_keys_match_loss。
SUPERVISED_KEYS = ["P_plant", "w_chiller", "w_chiller_on", "cold_back",
                   "cool_back", "cool_out", "tower_out",
                   "w_tower_dev", "w_coolpump_dev", "w_coldpump_dev"]

# head -> 是否应当收到梯度。False 的必须写明理由。
EXPECT_GRAD = {
    "head_a": True,        # approx -> tower_out -> k4
    "head_d": True,        # cool_dt -> cool_out -> k4（默认 cool_dt_mode="free"）
    "head_delta": True,    # delta -> cool_back -> k4
    "head_w": True,        # W -> P_plant / 逐台 -> k1/k2/k3
    # 默认 dt_evap_mode="soft" 下 eta **不经观测损失**，只经 L-soft-B 的
    # 软一致项 lam_evap_bal。见 test_eta_gets_gradient_from_soft_balance。
    "head_eta": False,
    "head_q": False,       # 仅 q_mode="free" 使用，from_w 下不接入
}


def _backward_supervised():
    torch.manual_seed(0)
    model = WorldModel(ModelConfig(), N_DEV, F)
    type_id = torch.tensor([0] + [1] * 7 + [2] * 20 + [3] * 7 + [4] * 8)
    batch = {"seq_x": torch.randn(B, W + H, N, F),
             "seq_avail": torch.ones(B, W + H, N, F),
             "seq_raw": torch.rand(B, W + H, N, F) * 10 + 5}
    out = model.rollout(batch, desc=torch.randn(N, 32), stat_rel=torch.randn(N, N, 5),
                        type_id=type_id, site_ctx=torch.zeros(B, 8), H=H, W=W)
    loss = torch.zeros(())
    for p in out["preds"]:
        for k in SUPERVISED_KEYS:
            if k in p:
                loss = loss + p[k].square().mean()
    loss.backward()
    return model


def _grad_norm(mod) -> float:
    """Module 或裸 Parameter 都能查 —— 物理参数不都挂在 Module 上。"""
    ps = [mod] if isinstance(mod, torch.nn.Parameter) else list(mod.parameters())
    return sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)


@pytest.mark.parametrize("head", sorted(k for k, v in EXPECT_GRAD.items() if v))
def test_head_receives_gradient(head):
    """应被约束的 head 必须收到非零梯度，否则它的输出是初始化值。"""
    m = _backward_supervised()
    mod = getattr(m.decoder, head)
    gn = _grad_norm(mod)
    assert gn > 1e-12, (
        f"{head} 梯度为 {gn:.3e} —— 该物理量未被任何数据约束，"
        f"报出的数值是初始化值，不可作为物理验证的证据")


def test_power_heads_receive_gradient():
    m = _backward_supervised()
    for fam, mod in m.decoder.head_p.items():
        assert _grad_norm(mod) > 1e-12, f"head_p[{fam}] 无梯度"


def test_unused_head_has_no_gradient():
    """q_mode='from_w' 下 head_q 不应接入 —— 若它有梯度说明走错了分支。"""
    m = _backward_supervised()
    assert _grad_norm(m.decoder.head_q) == 0.0


@pytest.mark.parametrize("head", sorted(k for k, v in EXPECT_GRAD.items() if not v))
def test_false_entries_really_have_no_gradient(head):
    """标 `expect_grad=False` 的 head 必须**确实**没有梯度。

    补的是本测试文件自己的一个洞：原先 False 的条目从不被检查，
    于是把一个仍在使用的 head 误标成 False，整套测试照样全绿 ——
    与 #24 是同一类静默失效，只不过藏在测试里。
    实测踩过：默认模式从 tied 改回 free 后 head_d 又被启用，标记却还是 False。
    """
    m = _backward_supervised()
    assert _grad_norm(getattr(m.decoder, head)) == 0.0, (
        f"{head} 标记为不使用，实际却有梯度 —— 标记与代码已经漂移")


def test_supervised_keys_match_loss():
    """本文件的 SUPERVISED_KEYS 必须与 losses.py 实际引用的键一致。

    否则「梯度检查」本身会失效：清单里多写了没进损失的键，就会给出
    虚假的「有梯度」结论。
    """
    src = (ROOT / "physwm" / "train" / "losses.py").read_text(encoding="utf-8")
    import re
    keys = set(re.findall(r'pred\["([a-z_]+)"\]', src))
    keys |= set(re.findall(r'pred\[f"(w_\{[a-z]+\}_dev)"\]', src))
    # 动态构造的族级键
    for fam in ("tower", "coolpump", "coldpump"):
        if 'w_{f}_dev' in src:
            keys.add(f"w_{fam}_dev")
    keys.discard("w_{f}_dev")
    # w_chiller_on 由 FAM_ORDER 分支间接引用
    if "w_chiller_on" in src:
        keys.add("w_chiller_on")
    missing = keys - set(SUPERVISED_KEYS)
    assert not missing, (
        f"losses.py 引用了本测试未覆盖的键 {missing} —— "
        f"梯度检查会漏掉经这些键传播的通路")


# ---------------------------------------------------------------------------
# 修 #24 时引入的**新**风险：把 eta 的梯度接回来，很容易顺手造出新的不可辨识对。
# 下面两条守卫的就是这个 —— 它们不查梯度，查的是「梯度接回来的方式对不对」。
# ---------------------------------------------------------------------------

def test_mcp_is_per_chiller_constant_not_per_sample():
    """m·cp 必须是每机常数，不能是逐样本预测。

    若 mcp 逐样本自由预测，`dt_evap = q_evap/mcp = W·COP_c·eta/mcp` 中
    (eta, mcp) 可同比缩放而不改变任何被监督的量 —— 与 §4.3 那个 (Q, eta)
    是同一个不可辨识对。届时 eta **有梯度但仍无意义**，
    test_head_receives_gradient 会全绿地放它过去。
    """
    model = WorldModel(ModelConfig(), N_DEV, F)
    # mcp 现在由设备描述符生成（M2 跨站迁移的要求，见 tests/test_cross_site.py），
    # 但**必须仍是每机一个常数**：desc 是逐设备、不随时间变的，故输出无 batch 维。
    desc_ch = torch.randn(N_DEV["chiller"], 32)
    z_ch = torch.randn(4, N_DEV["chiller"], model.cfg.d)
    ev, cd = model.decoder._mcp_raw(desc_ch, z_ch)
    assert ev.dim() == 1 and ev.shape[0] == N_DEV["chiller"], (
        f"_mcp_raw 输出形状 {tuple(ev.shape)}，期望 ({N_DEV['chiller']},)。"
        f"带 batch 维即逐样本 mcp，会让 (eta, mcp) 退回不可辨识")
    assert cd.shape == ev.shape
    assert not hasattr(model.decoder, "head_mcp"), (
        "出现了 head_mcp —— mcp 若由 head 逐样本预测，eta 的可辨识性即失效")
    # 它不能依赖 z（z 逐样本逐时刻变），否则等价于逐样本预测
    ev2, _ = model.decoder._mcp_raw(desc_ch, torch.randn_like(z_ch))
    assert torch.allclose(ev, ev2), "mcp 随 z 变化 —— 实际是逐样本的，可辨识性失效"


def test_dt_evap_lands_in_physical_range_at_init():
    """初始化时 dt_evap 必须落在物理量级，守 §13 #6 的量纲失配。

    #6 的原形是把 m·cp 硬编码成 `q_scale/dt_scale`：q_evap 从 ~500 变 ~3700 后
    算出 dT = 29 K（真值 5 K），k4 被搞坏且**不报任何错**。
    现在 mcp_scale 由数据推出（w_scale·COP_carnot·eta_mid/dt_evap_scale），
    这条测试保证它确实被算对了、而不是退回某个默认常数。
    """
    from tests.test_invariants import YB3_SCALES, _mk_inputs
    model = WorldModel(ModelConfig(), N_DEV, F)
    model.decoder.set_scales(**YB3_SCALES)
    z, alpha, phys = _mk_inputs(model, N_DEV, B=16)
    with torch.no_grad():
        o = model.decoder(z, alpha, phys)
    on = phys["on_ch"] > 0.5
    dt = o["dt_evap"][on] if on.any() else o["dt_evap"].reshape(-1)
    med = float(dt.median())
    assert torch.isfinite(dt).all(), "dt_evap 出现非有限值"
    assert (dt > 0).all(), "dt_evap 必须严格为正（蒸发侧取热）"
    assert 1.0 <= med <= 15.0, (
        f"dt_evap 中位 {med:.2f} K 落在物理量级之外（yb3 实测 5.0 K）。"
        f"最可能的原因是 mcp_scale 没有按数据推出 —— 这正是 #6 的复发形态")


def test_free_ablation_arm_reproduces_the_bug():
    """`dt_evap_mode="free"` 必须**确实**切断 eta 的梯度。

    这个臂是 §13 #24 的病灶本身，保留下来量化「把 dt_evap 接回 q_evap」的精度代价。
    若哪天它不再复现该 bug，说明对照臂坏了 —— 那时拿它做的一切对比都无效，
    而这件事不会以任何其他方式暴露出来。
    """
    torch.manual_seed(0)
    # **两个模式都要关掉。** 冷凝侧接上以后 eta 多了一条独立通路
    # （cool_dt = q_cond/mcp_cool 里含 q_evap），只关蒸发侧已不足以切断它 ——
    # 这本身就是「接冷凝侧确实提供了第二条约束」的直接证据。
    model = WorldModel(ModelConfig(dt_evap_mode="free", cool_dt_mode="free"),
                       N_DEV, F)
    type_id = torch.tensor([0] + [1] * 7 + [2] * 20 + [3] * 7 + [4] * 8)
    batch = {"seq_x": torch.randn(B, W + H, N, F),
             "seq_avail": torch.ones(B, W + H, N, F),
             "seq_raw": torch.rand(B, W + H, N, F) * 10 + 5}
    out = model.rollout(batch, desc=torch.randn(N, 32), stat_rel=torch.randn(N, N, 5),
                        type_id=type_id, site_ctx=torch.zeros(B, 8), H=H, W=W)
    loss = torch.zeros(())
    for p in out["preds"]:
        for k in SUPERVISED_KEYS:
            if k in p:
                loss = loss + p[k].square().mean()
    loss.backward()
    assert _grad_norm(model.decoder.head_eta) < 1e-12, (
        "free 臂居然给了 eta 梯度 —— 对照臂已不再复现 #24，用它做的对比无效")
    assert _grad_norm(model.decoder.head_dt_ev) > 1e-12, "free 臂的 dt_evap head 无梯度"


def test_head_d_unused_when_condenser_tied():
    """tied 臂里 head_d 必须无梯度（cool_dt 改为导出）。

    与 EXPECT_GRAD 里 head_d=True（默认 free 臂有梯度）配成一对：
    两头都断言，才说明它是**换了位置**而不是坏了。
    """
    torch.manual_seed(0)
    m_free = WorldModel(ModelConfig(cool_dt_mode="tied"), N_DEV, F)
    type_id = torch.tensor([0] + [1] * 7 + [2] * 20 + [3] * 7 + [4] * 8)
    batch = {"seq_x": torch.randn(B, W + H, N, F),
             "seq_avail": torch.ones(B, W + H, N, F),
             "seq_raw": torch.rand(B, W + H, N, F) * 10 + 5}
    out = m_free.rollout(batch, desc=torch.randn(N, 32), stat_rel=torch.randn(N, N, 5),
                         type_id=type_id, site_ctx=torch.zeros(B, 8), H=H, W=W)
    loss = torch.zeros(())
    for p in out["preds"]:
        for k in SUPERVISED_KEYS:
            if k in p:
                loss = loss + p[k].square().mean()
    loss.backward()
    assert _grad_norm(m_free.decoder.head_d) == 0.0, (
        "tied 臂里 head_d 仍有梯度 —— cool_dt 没有真的改成导出")


def test_condenser_tie_gives_eta_a_second_path():
    """接冷凝侧后，即使蒸发侧断开，eta 仍须有梯度。

    这正是「冷凝侧提供了第二条独立约束」的可执行证据 ——
    eta 尺度可辨识就建立在这条通路上。
    """
    torch.manual_seed(0)
    m = WorldModel(ModelConfig(dt_evap_mode="free", cool_dt_mode="tied"), N_DEV, F)
    type_id = torch.tensor([0] + [1] * 7 + [2] * 20 + [3] * 7 + [4] * 8)
    batch = {"seq_x": torch.randn(B, W + H, N, F),
             "seq_avail": torch.ones(B, W + H, N, F),
             "seq_raw": torch.rand(B, W + H, N, F) * 10 + 5}
    out = m.rollout(batch, desc=torch.randn(N, 32), stat_rel=torch.randn(N, N, 5),
                    type_id=type_id, site_ctx=torch.zeros(B, 8), H=H, W=W)
    loss = torch.zeros(())
    for p in out["preds"]:
        for k in SUPERVISED_KEYS:
            if k in p:
                loss = loss + p[k].square().mean()
    loss.backward()
    assert _grad_norm(m.decoder.head_eta) > 1e-12, (
        "冷凝侧未能给 eta 提供梯度 —— 不动点或 mcp_cool 的接线断了，"
        "eta 尺度将退回不可辨识")
    assert _grad_norm(m.decoder.mcp_net) > 1e-12, "mcp_net 无梯度（冷凝侧接线断了）"


def test_k_mcp_is_not_trainable():
    """`k_mcp` 必须是 buffer 而非 Parameter。

    若它可训，`dt_evap = W·COP_c·eta/(k_mcp·flow)` 里 (eta, k_mcp) 只以比值出现，
    又是一条平坦方向 —— eta 会重演 #29 的滑到界上。
    尺度这一维数据决定不了，就不要给模型一个假装能决定它的参数。
    """
    m = WorldModel(ModelConfig(), N_DEV, F)
    names = {n for n, _ in m.decoder.named_parameters()}
    assert "k_mcp" not in names, "k_mcp 变成了可训练参数 —— eta 的平坦方向回来了"
    assert isinstance(m.decoder.k_mcp, torch.Tensor)
    assert not m.decoder.k_mcp.requires_grad


def test_eta_level_prior_penalises_level_not_shape():
    """先验只压 eta 的**水平**，不压形状。

    若写成逐样本罚，eta 会被拉平成常数，等于把好不容易学到的形状又抹掉。
    """
    from physwm.train.losses import LossWeights, compute_loss

    torch.manual_seed(0)
    lg = torch.randn(B, 7, requires_grad=True)

    def _loss(shift):
        preds = [{"P_plant": torch.zeros(B), "eta_logit": lg + shift}]
        out = {"preds": preds}
        batch = {"seq_raw": torch.zeros(B, W + 1, N, F),
                 "seq_avail": torch.ones(B, W + 1, N, F),
                 "steady": torch.ones(B, 1),
                 "P_plant": torch.zeros(B, 1), "P_fam": torch.zeros(B, 1, 4)}

        class _Obs:
            def __call__(self, *a, **k):
                return {c: torch.zeros(()) for c in ("k1", "k2", "k3", "k4", "k5")}
        w = LossWeights(lam_eta_level=1.0, lam_lat=0.0)
        return compute_loss(None, out, batch, _Obs(), w, W)[0]

    # 整体平移水平 -> 惩罚必须变
    assert float(_loss(2.0)) > float(_loss(0.0)) + 1e-6, "水平平移未被惩罚"
    # 同一水平下改变形状（均值不变）-> 惩罚必须几乎不变
    centred = lg - lg.mean()
    preds = [{"P_plant": torch.zeros(B), "eta_logit": centred}]
    preds2 = [{"P_plant": torch.zeros(B), "eta_logit": centred * 3.0}]
    batch = {"seq_raw": torch.zeros(B, W + 1, N, F),
             "seq_avail": torch.ones(B, W + 1, N, F), "steady": torch.ones(B, 1),
             "P_plant": torch.zeros(B, 1), "P_fam": torch.zeros(B, 1, 4)}

    class _Obs:
        def __call__(self, *a, **k):
            return {c: torch.zeros(()) for c in ("k1", "k2", "k3", "k4", "k5")}
    w = LossWeights(lam_eta_level=1.0, lam_lat=0.0)
    l1 = float(compute_loss(None, {"preds": preds}, batch, _Obs(), w, W)[0])
    l2 = float(compute_loss(None, {"preds": preds2}, batch, _Obs(), w, W)[0])
    assert abs(l1 - l2) < 1e-8, f"形状被惩罚了：{l1} vs {l2}（先验应只作用于水平）"


def test_eta_gets_gradient_from_soft_balance():
    """soft 臂里 eta 的梯度全部来自 L-soft-B 的软一致项。

    与 EXPECT_GRAD 里 head_eta=False 配成一对：观测损失给不了它梯度，
    物理一致项必须给得了。少了任何一头，#24 都会以新形态复活。
    """
    from physwm.train.losses import LossWeights, compute_loss

    torch.manual_seed(0)
    model = WorldModel(ModelConfig(), N_DEV, F)
    type_id = torch.tensor([0] + [1] * 7 + [2] * 20 + [3] * 7 + [4] * 8)
    batch = {"seq_x": torch.randn(B, W + H, N, F),
             "seq_avail": torch.ones(B, W + H, N, F),
             "seq_raw": torch.rand(B, W + H, N, F) * 10 + 5,
             "steady": torch.ones(B, H),
             "P_plant": torch.rand(B, H) * 1000,
             "P_fam": torch.rand(B, H, 4) * 100}
    out = model.rollout(batch, desc=torch.randn(N, 32), stat_rel=torch.randn(N, N, 5),
                        type_id=type_id, site_ctx=torch.zeros(B, 8), H=H, W=W)

    class _Obs:
        def __call__(self, *a, **k):
            return {c: torch.zeros(()) for c in ("k1", "k2", "k3", "k4", "k5")}
    w = LossWeights(lam_evap_bal=1.0, lam_lat=0.0, lam_eta_level=0.0)
    loss, _ = compute_loss(model, out, batch, _Obs(), w, W)
    loss.backward()
    assert _grad_norm(model.decoder.head_eta) > 1e-12, (
        "软一致项没能给 eta 梯度 —— dt_evap_bal 的接线断了，#24 以新形态复活")


def test_soft_mode_without_balance_weight_raises():
    """soft 模式 + lam_evap_bal=0 必须报错，不能静默跑。

    这个组合让 eta 完全没有梯度，而训练日志、硬约束、物理量回读**都看不出来**。
    """
    import pytest as _pt
    from physwm.train.losses import LossWeights, compute_loss

    model = WorldModel(ModelConfig(), N_DEV, F)
    out = {"preds": [{"P_plant": torch.zeros(B)}]}
    batch = {"seq_raw": torch.zeros(B, W + 1, N, F),
             "seq_avail": torch.ones(B, W + 1, N, F), "steady": torch.ones(B, 1),
             "P_plant": torch.zeros(B, 1), "P_fam": torch.zeros(B, 1, 4)}

    class _Obs:
        def __call__(self, *a, **k):
            return {c: torch.zeros(()) for c in ("k1", "k2", "k3", "k4", "k5")}
    with _pt.raises(RuntimeError, match="lam_evap_bal"):
        compute_loss(model, out, batch, _Obs(), LossWeights(lam_evap_bal=0.0), W)


def test_soft_mode_guard_does_not_fire_under_no_grad():
    """守卫只在训练期生效 —— 评测期把正则权重清零是合法用法。

    多站评测必须把各站不可比的正则项清零才能横比验证损失。
    第一版守卫漏了这个边界条件，直接把自己的评测拦下了。
    """
    from physwm.train.losses import LossWeights, compute_loss

    model = WorldModel(ModelConfig(), N_DEV, F)
    out = {"preds": [{"P_plant": torch.zeros(B)}]}
    batch = {"seq_raw": torch.zeros(B, W + 1, N, F),
             "seq_avail": torch.ones(B, W + 1, N, F), "steady": torch.ones(B, 1),
             "P_plant": torch.zeros(B, 1), "P_fam": torch.zeros(B, 1, 4)}

    class _Obs:
        def __call__(self, *a, **k):
            return {c: torch.zeros(()) for c in ("k1", "k2", "k3", "k4", "k5")}
    with torch.no_grad():
        compute_loss(model, out, batch, _Obs(), LossWeights(lam_evap_bal=0.0), W)
