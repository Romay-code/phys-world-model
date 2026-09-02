"""方向一致性：`dir_vr` 在 rollout 上的推广（设计文档 §5.3，P4 交付件）。

对每个起点与每个动作维：
    轨迹 A：动作序列不变            -> 推演 H 步 -> y_A(h)
    轨迹 B：a[j] += Δ（其余不变）   -> 推演 H 步 -> y_B(h)
    比较 sign(y_B(h) - y_A(h)) 与该维的物理先验符号，统计违例比例。

两条实现上的要害：

1. **只改推演段的动作维。** 历史窗必须逐位保留，否则 z_{t0} 跟着变了，
   比较的就不是「同一 z 下换动作」。`WorldModel.counterfactual` 已经把这条
   写进契约，本模块只负责构造合法的 intervene。

2. **归一化值与原始值必须同步改。** `seq_x` 进编码器、`seq_raw` 进物理解码器，
   只改一个会让两条通路看到不同的世界。Normalizer 是逐 (token, field) 的线性
   变换 `xn = (x - center)/scale`，故原始扰动 Δ 对应归一化扰动 Δ/scale[n,k]。

**为什么先验符号表不挂在 P_plant 上（对设计文档 §5.3 的更正）。**
§5.3 原文写的是拿 `P_plant` 比先验符号。对**塔频与泵频这是物理上错误的**：
塔频↑ 使塔功率↑（三次方律）但冷凝温度↓ 从而冷机功率↓，净效应在最优点两侧变号
—— 冷站节能优化之所以是个非平凡问题，正是因为这条曲线非单调。给它安一个单调
先验，等于把「优化问题存在解」这件事本身判成违例。

故本模块按**分量**挂先验：塔频挂在逼近度/塔功率/冷凝进水温上，泵频挂在自身功率上。
只有 `cold_out_temp` 对 `P_plant` 是有明确单调先验的（供水温↑ -> 蒸发温度↑ ->
COP↑ -> 冷机功率↓，泵塔基本不受影响）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..data import schema as S


@dataclass
class DirSpec:
    """一个动作维的扰动与它的物理先验。"""

    name: str
    family: str
    fld: str
    delta: float                       # 原始量纲的扰动幅度（Hz / K）
    targets: dict[str, int]            # 预测键 -> 期望符号 (+1 / -1)
    why: str                           # 物理依据，必须写
    on_only: bool = True               # 只扰动开机设备（停机设备加频率无意义）
    clip: tuple[float, float] | None = None   # 扰动后裁到实测分布内（G3）


def default_specs() -> list[DirSpec]:
    """默认先验表。每一条都要能用一句物理讲清楚，讲不清的不要放进来。"""
    return [
        DirSpec("塔频+1Hz", "tower", "frequency", 1.0,
                {"approx": -1, "w_tower_dev": +1, "cool_out": -1},
                why="风量↑ -> 逼近度↓、塔功率↑（三次方律）、冷凝进水温↓；"
                    "**不挂 P_plant**：净效应在最优点两侧变号"),
        DirSpec("冷却泵频+1Hz", "coolpump", "frequency", 1.0,
                {"w_coolpump_dev": +1},
                why="三次方律。同样不挂 P_plant"),
        DirSpec("冷冻泵频+1Hz", "coldpump", "frequency", 1.0,
                {"w_coldpump_dev": +1},
                why="三次方律"),
        DirSpec("冷冻供水温+0.5K", "chiller", "cold_out_temp", 0.5,
                {"w_chiller": -1, "P_plant": -1},
                why="蒸发温度↑ -> COP_carnot↑ -> 同负荷下冷机功率↓，泵塔基本不变。"
                    "G3：cold_out_temp 暂按准动作，幅度限制在实测分布内",
                clip=S.RANGE_RULES["cold_out_temp"]),
    ]


def _token_rows(sch, family: str) -> list[int]:
    return [i for i, (f, _) in enumerate(sch.token_index) if f == family]


def make_intervene(sch, norm_scale: torch.Tensor, spec: DirSpec, type_id):
    """构造 counterfactual() 要的 intervene(seq_x, seq_raw, W)。"""
    rows = _token_rows(sch, spec.family)
    k = S.CANON_FIELDS[spec.family].index(spec.fld)
    k_on = S.CANON_FIELDS[spec.family].index("on")
    sc = norm_scale[rows, k]                      # [n_dev]

    def intervene(seq_x, seq_raw, W):
        # 只动推演段 [W:]，历史窗逐位保留
        raw_seg = seq_raw[:, W:, rows, k]                       # [B,H,n_dev]
        if spec.on_only:
            m = (seq_raw[:, W:, rows, k_on] > 0.5).to(raw_seg.dtype)
        else:
            m = torch.ones_like(raw_seg)
        new = raw_seg + spec.delta * m
        if spec.clip is not None:
            lo, hi = spec.clip
            new = new.clamp(lo, hi)
        d_raw = new - raw_seg                                    # 实际生效的增量
        seq_raw[:, W:, rows, k] = new
        seq_x[:, W:, rows, k] = seq_x[:, W:, rows, k] + d_raw / sc
        return seq_x, seq_raw

    return intervene


def _stack(preds: list[dict], key: str) -> torch.Tensor | None:
    if key not in preds[0]:
        return None
    return torch.stack([p[key] for p in preds], 1)      # [B,H,...]


@torch.no_grad()
def rollout_direction_check(model, ds, ctx, *, H: int, device, sch,
                            norm_scale: torch.Tensor,
                            specs: list[DirSpec] | None = None,
                            batch_size: int = 32, max_batches: int = 20,
                            tol: float = 0.0, scales: dict | None = None) -> dict:
    """逐 h 的方向违例率。返回 {spec.name: {target: {"per_h": [...], ...}}}。"""
    specs = specs or default_specs()
    model.eval()
    ds.set_H(H)
    ns = torch.as_tensor(norm_scale, dtype=torch.float32, device=device)

    acc: dict[str, dict[str, list]] = {
        s.name: {t: [] for t in s.targets} for s in specs}

    kw = dict(desc=ctx.desc, stat_rel=ctx.stat_rel, type_id=ctx.type_id,
              W=ctx.W, key_mask=ctx.key_mask, scales=scales)
    for i, b in enumerate(ds.epoch(batch_size, shuffle=False, drop_last=False)):
        if i >= max_batches:
            break
        B = b["P_plant"].shape[0]
        sctx = ctx.ctx(B, device)
        base = model.imagine(b, site_ctx=sctx, H=H, **kw)["preds"]
        for s in specs:
            fn = make_intervene(sch, ns, s, ctx.type_id)
            pert = model.counterfactual(b, fn, site_ctx=sctx, H=H, **kw)["preds"]
            for tgt, sign in s.targets.items():
                a, p = _stack(base, tgt), _stack(pert, tgt)
                if a is None:
                    continue
                d = (p - a).reshape(B, H, -1)
                # 停机设备两条轨迹都是 0，差恒为 0，会把违例率稀释成好看的数字。
                # 只统计基线上确有量值的位置。
                live = (a.reshape(B, H, -1).abs() > 1e-6)
                bad = (d < -tol) if sign > 0 else (d > tol)
                acc[s.name][tgt].append(
                    torch.stack([(bad & live).sum((0, 2)).float(),
                                 live.sum((0, 2)).float()], 0).cpu().numpy())

    out: dict = {}
    for s in specs:
        out[s.name] = {"why": s.why, "delta": s.delta,
                       "field": f"{s.family}.{s.fld}", "targets": {}}
        for tgt, sign in s.targets.items():
            chunks = acc[s.name][tgt]
            if not chunks:
                continue
            z = np.sum(np.stack(chunks, 0), 0)          # [2, H]
            n_bad, n_tot = z[0], np.maximum(z[1], 1.0)
            per_h = (n_bad / n_tot).tolist()
            out[s.name]["targets"][tgt] = {
                "expect_sign": sign,
                "per_h": per_h,
                "n_eval": float(z[1].sum()),
                "dir_vr_1": per_h[0] if per_h else float("nan"),
                "dir_vr_max": max(per_h) if per_h else float("nan"),
            }
            for h in (12, 24, 36, 48, 96):
                if h <= len(per_h):
                    out[s.name]["targets"][tgt][f"dir_vr_{h}"] = per_h[h - 1]
    return out


def format_direction(rep: dict, at: int = 36) -> str:
    """一行一个 (动作维, 目标)。门限：dir_vr(36) <= 0.10。"""
    lines = [f"{'动作维':<16}{'目标':<16}{'符号':>4}{'h=1':>8}{f'h={at}':>8}"
             f"{'最差':>8}   判定"]
    for name, d in rep.items():
        for tgt, t in d["targets"].items():
            v = t.get(f"dir_vr_{at}", float("nan"))
            ok = "OK" if v <= 0.10 else "**超门限 0.10**"
            lines.append(f"{name:<16}{tgt:<16}{t['expect_sign']:>+4}"
                         f"{t['dir_vr_1']:>8.3f}{v:>8.3f}{t['dir_vr_max']:>8.3f}   {ok}")
    return "\n".join(lines)
