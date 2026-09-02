"""站点量纲推断 —— 从该站自己的训练数据算出各物理量的量级。

**这些是站点属性，不是模型属性。** 实测 14 站：`w_scale` 跨 6.9 倍
（178.5 tx ~ 1229.0 yb_低温）、`k_mcp` 跨 38.5 倍、`q_scale` 跨 22.2 倍。
它们不进模型权重、不参与跨站迁移（M2 迁移的是无量纲的结构），
部署新站时由该站历史数据现场估 —— 「零样本」指不微调，不是没有该站数据。

原先住在 `experiments/smoke_yb3.py`，多站训练需要它们而库不该反向依赖实验脚本，
故移来此处；那两个脚本改为从这里导入，函数体逐字未改。
"""
from __future__ import annotations

import numpy as np

from . import schema as S
from ..model.decoder import ETA_MAX, ETA_MIN


def infer_scales(sd, row_mask) -> dict[str, float]:
    """从训练集统计量纲。塔/泵功率按族的中位数分摊到每台。"""
    e = sd.extra
    out: dict[str, float] = {}
    for fam, n in (("tower", sd.sch.n_dev["tower"]),
                   ("coolpump", sd.sch.n_dev["coolpump"]),
                   ("coldpump", sd.sch.n_dev["coldpump"])):
        p = e[f"power_{fam}"][row_mask]
        out[f"p_scale_{fam}"] = (float(np.median(p[p > 0])) / max(n, 1)
                                 if (p > 0).any() else 50.0)

    # q_scale 仅在 q_mode="free" 下用：站负荷 / 中位开机台数
    k_load = S.CANON_FIELDS["plant"].index("load")
    load = sd.x[row_mask, 0, k_load]
    k_on = S.CANON_FIELDS["chiller"].index("on")
    ch = [i for i, (f, _) in enumerate(sd.sch.token_index) if f == "chiller"]
    n_on = sd.x[row_mask][:, ch, k_on].sum(-1)
    n_on_med = max(float(np.median(n_on[n_on > 0])) if (n_on > 0).any() else 1.0, 1.0)
    out["q_scale"] = float(np.median(load[load > 0])) / n_on_med
    return out


def infer_w_scale(sd, row_mask) -> float:
    """单台冷机功率的量纲，用于 head_w 的输出缩放。

    取「开机时冷机族总功率 / 开机台数」的中位。网络内部保持 O(1)，
    物理量纲由这个 buffer 承载（不是可训练参数 —— 它是量纲换算，不该被梯度改）。
    """
    e = sd.extra
    k_on = S.CANON_FIELDS["chiller"].index("on")
    ch = [i for i, (f, _) in enumerate(sd.sch.token_index) if f == "chiller"]
    n_on = sd.x[row_mask][:, ch, k_on].sum(-1)
    w = e["power_chiller"][row_mask]
    m = (n_on > 0) & (w > 0)
    return float(np.median(w[m] / n_on[m])) if m.sum() > 32 else 500.0


def infer_dt_evap_scale(sd, row_mask) -> float:
    """蒸发侧温差 cold_back - cold_out 的量纲（仅开机且两端有效）。yb3 实测 p50=5.0 K。"""
    K = S.CANON_FIELDS["chiller"]
    ch = [i for i, (f, _) in enumerate(sd.sch.token_index) if f == "chiller"]
    x, av = sd.x[row_mask][:, ch], sd.avail[row_mask][:, ch].astype(bool)
    on = x[:, :, K.index("on")] > 0.5
    co, cb = K.index("cold_out_temp"), K.index("cold_back_temp")
    m = on & av[:, :, co] & av[:, :, cb]
    d = (x[:, :, cb] - x[:, :, co])[m]
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size > 32 else 5.0


def infer_k_mcp(sd, row_mask, w_scale: float, dt_evap_scale: float) -> float:
    """流量驱动的 m·cp 系数：`m·cp = k_mcp · Σ(开机冷冻泵频) / 开机冷机数`。

    典型工况下应当有 `dt_evap = q_evap/mcp ≈ dt_evap_scale`，于是

        mcp_typ = w_scale · COP_carnot_typ · eta_mid / dt_evap_scale
        k_mcp   = mcp_typ · n_chiller_typ / flow_typ

    与 infer_mcp_scale 同理，eta 取物理区间中点 —— 它只决定这个 buffer 的取值。
    **k_mcp 不可训练**：若它与 eta 同时可训，二者只以比值出现，又是一条平坦方向。
    """
    K = S.CANON_FIELDS["chiller"]
    KP = S.CANON_FIELDS["coldpump"]
    ch = [i for i, (f, _) in enumerate(sd.sch.token_index) if f == "chiller"]
    cp = [i for i, (f, _) in enumerate(sd.sch.token_index) if f == "coldpump"]
    if not cp:
        return 15.0
    x = sd.x[row_mask]
    on_ch = x[:, ch, K.index("on")] > 0.5
    n_on = on_ch.sum(1)
    p_on = x[:, cp, KP.index("on")] > 0.5
    p_fr = np.nan_to_num(x[:, cp, KP.index("frequency")])
    flow = (p_fr * p_on).sum(1)
    m = (n_on > 0) & (flow > 1)
    if m.sum() < 32:
        return 15.0
    # COP_carnot 典型值
    co, uo = K.index("cold_out_temp"), K.index("cool_out_temp")
    av = sd.avail[row_mask][:, ch].astype(bool)
    mm = on_ch & av[:, :, co] & av[:, :, uo]
    t_ev = x[:, ch, co][mm] + 273.15
    t_cd = np.maximum(x[:, ch, uo][mm] + 273.15, t_ev + 2.0)
    cop_c = t_ev / (t_cd - t_ev)
    cop_c = cop_c[np.isfinite(cop_c) & (cop_c > 0)]
    if cop_c.size <= 32:
        return 15.0
    eta_mid = 0.5 * (ETA_MIN + ETA_MAX)
    mcp_typ = float(w_scale) * float(np.median(cop_c)) * eta_mid / max(float(dt_evap_scale), 1e-3)
    ratio = float(np.median(n_on[m] / np.maximum(flow[m], 1e-6)))
    return max(mcp_typ * ratio, 1e-6)


def infer_mcp_scale(sd, row_mask, w_scale: float, dt_evap_scale: float) -> float:
    """蒸发侧 m·cp 的量纲 [kW/K]，用于 decoder 的 log_mcp_raw 缩放。

    yb3 没有冷冻流量，m·cp 无法直接测。但它可以由**三个都能测的量**定出来：

        Q_evap = W · COP_carnot · eta        （W 实测、COP_carnot 由实测温度算）
        m·cp   = Q_evap / ΔT_evap            （ΔT_evap 由实测两端温差算）

    eta 取物理区间中点 —— 它只决定这个 buffer 的**初始点**，训练中由
    log_mcp_raw 自行偏离，不是硬编码的物理假设。这样初始化时
    dt_evap = q_evap/mcp ≈ dt_evap_scale，避开 §13 #6 的量纲失配。
    """
    K = S.CANON_FIELDS["chiller"]
    ch = [i for i, (f, _) in enumerate(sd.sch.token_index) if f == "chiller"]
    x, av = sd.x[row_mask][:, ch], sd.avail[row_mask][:, ch].astype(bool)
    on = x[:, :, K.index("on")] > 0.5
    co, uo = K.index("cold_out_temp"), K.index("cool_out_temp")
    m = on & av[:, :, co] & av[:, :, uo]
    t_ev = x[:, :, co][m] + 273.15
    t_cd = np.maximum(x[:, :, uo][m] + 273.15, t_ev + 2.0)
    cop_c = t_ev / (t_cd - t_ev)
    cop_c = cop_c[np.isfinite(cop_c) & (cop_c > 0)]
    if cop_c.size <= 32:
        return 700.0
    eta_mid = 0.5 * (ETA_MIN + ETA_MAX)
    q_typ = float(w_scale) * float(np.median(cop_c)) * eta_mid
    return max(q_typ / max(float(dt_evap_scale), 1e-3), 1e-3)


def infer_mcp_cool_scale(sd, row_mask, w_scale: float) -> float:
    """冷凝侧 m·cp 的量纲 [kW/K]。

        Q_cond = W * (1 + COP_carnot * eta)          （能量守恒，全部可测/有先验）
        m·cp   = Q_cond / (cool_out - cool_back)     （两端温度实测）

    与 infer_mcp_scale 同理，eta 取区间中点只决定初始点，训练中由
    log_mcp_cool_raw 自行偏离。

    冷却侧温差直接从实测两端温度算（不用 decoder 的 dt_scale 默认值 5.0，
    那是个未经数据确认的常数）。
    """
    K = S.CANON_FIELDS["chiller"]
    ch = [i for i, (f, _) in enumerate(sd.sch.token_index) if f == "chiller"]
    x, av = sd.x[row_mask][:, ch], sd.avail[row_mask][:, ch].astype(bool)
    on = x[:, :, K.index("on")] > 0.5
    co, uo = K.index("cold_out_temp"), K.index("cool_out_temp")
    m = on & av[:, :, co] & av[:, :, uo]
    t_ev = x[:, :, co][m] + 273.15
    t_cd = np.maximum(x[:, :, uo][m] + 273.15, t_ev + 2.0)
    cop_c = t_ev / (t_cd - t_ev)
    cop_c = cop_c[np.isfinite(cop_c) & (cop_c > 0)]
    if cop_c.size <= 32:
        return 1500.0
    ub = K.index("cool_back_temp")
    m2 = on & av[:, :, uo] & av[:, :, ub]
    d = (x[:, :, uo] - x[:, :, ub])[m2]
    d = d[np.isfinite(d) & (d > 0)]
    cool_dt_med = float(np.median(d)) if d.size > 32 else 5.0
    eta_mid = 0.5 * (ETA_MIN + ETA_MAX)
    q_cond_typ = float(w_scale) * (1.0 + float(np.median(cop_c)) * eta_mid)
    return max(q_cond_typ / max(cool_dt_med, 1e-3), 1e-3)


def infer_site_scales(sd, row_mask) -> dict[str, float]:
    """一次算齐一个站点需要的全部量纲。多站训练的统一入口。"""
    sc = infer_scales(sd, row_mask)
    sc["w_scale"] = infer_w_scale(sd, row_mask)
    sc["dt_evap_scale"] = infer_dt_evap_scale(sd, row_mask)
    sc["mcp_scale"] = infer_mcp_scale(sd, row_mask, sc["w_scale"], sc["dt_evap_scale"])
    sc["mcp_cool_scale"] = infer_mcp_cool_scale(sd, row_mask, sc["w_scale"])
    sc["k_mcp"] = infer_k_mcp(sd, row_mask, sc["w_scale"], sc["dt_evap_scale"])
    return sc
