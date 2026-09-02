"""评测指标（设计文档 §5）。

`all_metrics` 与参考项目 `common/metrics.py` 保持逐位一致（含 dyMAE），
这样两个项目的数字可以直接对照。故意不跨项目 import —— 参考项目改动不应
影响本项目的历史结果（§12 G12）。
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# 与参考项目 common/metrics.py 对齐的实现（vendored, 冻结）
# --------------------------------------------------------------------------

def all_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    out = {
        "MAE": float(np.mean(abs_diff)),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "maxErr": float(np.max(abs_diff)),
        "P95AE": float(np.percentile(abs_diff, 95)),
        "P99AE": float(np.percentile(abs_diff, 99)),
    }
    nonzero = np.abs(y_true) > 1e-6
    out["MAPE"] = (float(np.mean(abs_diff[nonzero] / np.abs(y_true[nonzero])) * 100)
                   if nonzero.any() else float("nan"))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    out["R2"] = float(1.0 - np.sum(diff ** 2) / denom) if denom > 0 else float("nan")
    if len(y_true) > 1:
        out["dyMAE"] = float(np.mean(np.abs(np.diff(y_pred) - np.diff(y_true))))
    else:
        out["dyMAE"] = float("nan")
    return out


def format_metrics(rows: dict[str, dict[str, float]]) -> str:
    cols = ("MAE", "RMSE", "dyMAE", "R2", "P95AE", "maxErr")
    width = max([len(k) for k in rows] + [8])
    lines = ["name".ljust(width) + "".join(c.rjust(11) for c in cols)]
    lines.append("-" * len(lines[0]))
    for name, m in rows.items():
        lines.append(name.ljust(width) + "".join(f"{m.get(c, float('nan')):11.4g}"
                                                 for c in cols))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 有效推演步数 h*
# --------------------------------------------------------------------------

def per_step_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float]]:
    """y_* [M, H] -> 逐 h 的指标列表。"""
    return [all_metrics(y_true[:, h], y_pred[:, h]) for h in range(y_true.shape[1])]


def effective_horizon(per_h: list[dict[str, float]], *, r2_min: float = 0.80,
                      mae_ratio_max: float = 3.0,
                      slope_max: float = 1.0) -> dict[str, float]:
    """h* = max{ h : 对所有 k<=h, R2_k >= r2_min 且 MAE_k <= mae_ratio_max * MAE_1 }

    外加发散守门：log MAE 对 log h 的回归斜率 <= slope_max（次线性增长）。

    阈值 0.80 / 3.0 是设计文档 §5.1 的**建议值，尚待拍板**（§12 G9）。
    定下来之前所有 h* 数字都带这个前提。
    """
    if not per_h:
        return {"h_star": 0.0, "slope": float("nan"), "mae_1": float("nan")}
    mae1 = per_h[0]["MAE"]
    h_star = 0
    for h, m in enumerate(per_h, start=1):
        if m["R2"] >= r2_min and m["MAE"] <= mae_ratio_max * mae1:
            h_star = h
        else:
            break

    slope = float("nan")
    if h_star >= 2:
        hs = np.arange(1, h_star + 1, dtype=float)
        ms = np.array([per_h[i]["MAE"] for i in range(h_star)], dtype=float)
        ok = ms > 1e-9
        if ok.sum() >= 2:
            slope = float(np.polyfit(np.log(hs[ok]), np.log(ms[ok]), 1)[0])
            if slope > slope_max:
                # 增长快于线性 -> 判为发散，回退到斜率仍达标的最大前缀
                for hh in range(h_star, 1, -1):
                    s = float(np.polyfit(np.log(hs[:hh][ok[:hh]]),
                                         np.log(ms[:hh][ok[:hh]]), 1)[0])
                    if s <= slope_max:
                        h_star, slope = hh, s
                        break
                else:
                    h_star = 1
    return {"h_star": float(h_star), "slope": slope, "mae_1": float(mae1),
            "r2_min": r2_min, "mae_ratio_max": mae_ratio_max}


def effective_horizon_multi(per_h: list[dict[str, float]], *,
                           mae_abs_max: float | None = None,
                           r2_min: float = 0.80,
                           mae_ratio_max: float = 3.0,
                           slope_max: float = 1.0) -> dict[str, dict]:
    """同时按三种口径报 h*，并标出各自被哪一条判据卡住。

    **为什么要三个口径。** §5.1 的判据是 `R2 >= 0.80` 且 `MAE <= 3 x MAE_1`。
    P3-B 实测发现两件事（§13 #27）：

      1. `R2 >= 0.80` 从未生效 —— h=96 处最低的 seed 也有 0.9530，
         h* 完全由 MAE 那一条决定；
      2. 那一条的分母是模型**自己的** MAE_1，于是单步越准的模型长程标准越苛刻。
         实测单步最差的 seed（MAE_1=84.5）反而拿到比单步最好的 seed
         （59.6）更高的 h* —— 这正是 #23 已在「相对误差放大」上认定过的缺陷。

    G9 拍板前把三个口径一起算出来，换口径不必重训：

      A `r2_only`    只要 R2 >= r2_min。验收指标①/⑥的原文口径。
      B `self_ratio` 再加 MAE <= mae_ratio_max x MAE_1。设计文档 §5.1 现行建议值。
      C `abs_bar`    再加 MAE <= mae_abs_max（外部绝对阈，与模型自身表现无关）。
                     推荐取全站功率中位的一个比例，物理含义直接：
                     「推演误差不超过全站功率的 x%」。mae_abs_max=None 时不算这一档。

    每档额外返回：
      `binding`  卡住它的判据：'r2' / 'mae' / 'slope' / 'ceiling'
      `censored` h* 是否顶到了评测视野本身（True 时该数字的含义只是「>= H」，
                 是下界而非测得值 —— §13 #26）。
    """
    H = len(per_h)
    if H == 0:
        return {}
    mae1 = per_h[0]["MAE"]

    def _run(mae_cap: float | None) -> dict[str, float]:
        h_star, binding = 0, "ceiling"
        for h, m in enumerate(per_h, start=1):
            if m["R2"] < r2_min:
                binding = "r2"
                break
            if mae_cap is not None and m["MAE"] > mae_cap:
                binding = "mae"
                break
            h_star = h
        slope = float("nan")
        if h_star >= 2:
            hs = np.arange(1, h_star + 1, dtype=float)
            ms = np.array([per_h[i]["MAE"] for i in range(h_star)], dtype=float)
            ok = ms > 1e-9
            if ok.sum() >= 2:
                slope = float(np.polyfit(np.log(hs[ok]), np.log(ms[ok]), 1)[0])
                if slope > slope_max:
                    binding = "slope"
                    for hh in range(h_star, 1, -1):
                        s = float(np.polyfit(np.log(hs[:hh][ok[:hh]]),
                                             np.log(ms[:hh][ok[:hh]]), 1)[0])
                        if s <= slope_max:
                            h_star, slope = hh, s
                            break
                    else:
                        h_star = 1
        return {"h_star": float(h_star), "slope": slope,
                "binding": binding, "censored": bool(h_star >= H),
                "mae_cap": (float(mae_cap) if mae_cap is not None else None)}

    out = {"r2_only": _run(None),
           "self_ratio": _run(mae_ratio_max * mae1)}
    if mae_abs_max is not None:
        out["abs_bar"] = _run(float(mae_abs_max))
    out["_meta"] = {"mae_1": float(mae1), "eval_H": H, "r2_min": r2_min,
                    "mae_ratio_max": mae_ratio_max}
    return out


def error_amplification(tf_mae: float, fr_mae: float) -> dict[str, float]:
    """误差放大：teacher-forcing -> free-running 的退化。

    参考项目证明「隐向量级联比标量级联抗误差放大 10 倍」（yb3: A0 +250.9 vs B0 +25.3），
    但 B2 这一栏是空的。本项目从第一天起就纳入标准输出。
    """
    return {"tf_MAE": tf_mae, "fr_MAE": fr_mae,
            "abs_amp": fr_mae - tf_mae,
            "rel_amp": (fr_mae - tf_mae) / max(tf_mae, 1e-9)}


def dir_violation_rate(base: np.ndarray, perturbed: np.ndarray,
                       expect_sign: int, tol: float = 0.0) -> float:
    """方向违例率。base/perturbed [M] 或 [M,H]，expect_sign in {+1,-1}。"""
    d = np.asarray(perturbed) - np.asarray(base)
    if expect_sign > 0:
        bad = d < -tol
    else:
        bad = d > tol
    return float(bad.mean())
