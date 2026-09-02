"""同一份权重、同一个站，为什么换个划分口径 R² 差这么多？

少样本实验里撞见的：留出站 hx 的零样本 R²

    blocked（分块交错，各 split 覆盖全年）      **0.9335**
    chronological（时序切，test 是最后 20%）    **−2.3040**

模型是同一个、站是同一个，**只有「评哪些窗口」不同**。
若差异来自 test 段的工况本身（负荷更高、季节不同），那么 0.9335 是
「全年平均」而不是「任意时段」—— 这直接影响门限⑥该怎么表述。

本工具不训练，只回答三个问题：

  1. 两种口径的 test 段，**真值分布**差多少（均值/分位/月份）
  2. 模型在两段上的**偏差方向**（系统性低估还是随机误差）
  3. 若把 chronological 的 test 段单独按 R² 拆开看，是量级问题还是形状问题
     —— 复用 `diag_zeroshot_calib` 的三个数：R² / R²(仅比例) / R²(仿射)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402
from physwm.data.dataset import (GPUWindows, WindowSpec,  # noqa: E402
                                 enumerate_windows, split_windows)
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402
from physwm.train.multisite import build_sites  # noqa: E402
import dataclasses  # noqa: E402


@torch.no_grad()
def collect(model, site, windows, dev, H=8, max_batches=0):
    if len(windows) == 0:
        return np.array([]), np.array([]), np.array([])
    ds = GPUWindows(site.sd, windows, site.spec, site.norm, dev, H=H)
    ds.set_H(H)
    yt, yp, t0 = [], [], []
    for i, b in enumerate(ds.epoch(64, shuffle=False, drop_last=False)):
        # 0 = 不限。窗口按时间升序、epoch 不打乱，截断取到的是**最早**
        # 的一批窗口而非随机样本（§13 #56）—— 诊断工具同样会被它带偏。
        if max_batches and i >= max_batches:
            break
        o = model.rollout(b, desc=site.ctx.desc, stat_rel=site.ctx.stat_rel,
                          type_id=site.ctx.type_id,
                          site_ctx=site.ctx.ctx(b["P_plant"].shape[0], dev),
                          H=H, W=site.ctx.W, reanchor_p=0.0, scales=site.scales)
        yt.append(b["P_plant"][:, 0].cpu().numpy())
        yp.append(o["preds"][0]["P_plant"].cpu().numpy())
        t0.append(b["t0"].cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp), np.concatenate(t0)


def stats(yt, yp):
    if yt.size == 0:
        return {}
    yt, yp = yt.astype(np.float64), yp.astype(np.float64)
    ss = float(((yt - yt.mean()) ** 2).sum())
    r2 = 1 - float(((yt - yp) ** 2).sum()) / ss if ss > 0 else float("nan")
    k = float((yt * yp).sum() / max((yp * yp).sum(), 1e-12))
    r2k = 1 - float(((yt - k * yp) ** 2).sum()) / ss if ss > 0 else float("nan")
    c = float(np.corrcoef(yt, yp)[0, 1]) if yt.size > 2 else float("nan")
    return {"n": yt.size, "t_mean": yt.mean(), "t_std": yt.std(),
            "p_mean": yp.mean(), "p_std": yp.std(), "corr": c,
            "R2": r2, "R2_scaled": r2k, "R2_affine": c * c, "k": k,
            "bias": (yp - yt).mean()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sites", default="hx,pc3")
    ap.add_argument("--hold-out", default="hx,pc3")
    ap.add_argument("--H", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    spec = WindowSpec(W=16, H=48)
    want = tuple(x for x in a.sites.split(",") if x.strip())
    hold = tuple(x for x in a.hold_out.split(",") if x.strip())
    allsites = build_sites(ROOT / "data", dev, spec=spec, hold_out=hold,
                           artifacts=ROOT / "artifacts" / "multisite")
    big = max((s for s in allsites if not s.held_out), key=lambda s: s.sd.N)
    model = WorldModel(ModelConfig(delta=0.5), big.sd.sch.n_dev, big.sd.F).to(dev)
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(st.get("model", st) if isinstance(st, dict) else st)
    model.eval()

    for s in [x for x in allsites if x.name in want]:
        df = pd.read_csv(ROOT / "data" / f"{s.name}.csv", low_memory=False)
        ts = pd.to_datetime(df[s.sd.sch.time_col], errors="coerce")

        wins = enumerate_windows(s.sd, s.spec)
        chrono = split_windows(s.sd, wins,
                               dataclasses.replace(s.spec, split_mode="chronological"))
        cases = [("blocked   test", s.splits["test"]),
                 ("chrono    test", chrono["test"]),
                 ("chrono    train(适配段来源)", chrono["train"])]

        print(f"\n=== {s.name}  档 {s.tier} ===")
        print(f"{'口径':<26}{'n':>6}{'真值均值':>10}{'真值std':>9}{'预测均值':>10}"
              f"{'预测std':>9}{'偏差':>9}{'corr':>7}{'R²':>9}{'R²仿射':>8}   月份")
        print("-" * 122)
        for name, w in cases:
            yt, yp, t0 = collect(model, s, w, dev, a.H)
            st_ = stats(yt, yp)
            if not st_:
                print(f"{name:<26} (空)")
                continue
            mons = sorted(set(ts.iloc[t0].dt.to_period("M").astype(str)))
            ms = ",".join(m[2:] for m in mons)
            print(f"{name:<26}{st_['n']:>6}{st_['t_mean']:>10.0f}{st_['t_std']:>9.0f}"
                  f"{st_['p_mean']:>10.0f}{st_['p_std']:>9.0f}{st_['bias']:>9.0f}"
                  f"{st_['corr']:>7.3f}{st_['R2']:>9.3f}{st_['R2_affine']:>8.3f}"
                  f"   {ms[:40]}")
    print()
    print("读法：若 chrono-test 的真值均值明显高于 blocked-test，而预测跟不上（偏差为负），")
    print("      则 0.9335 是**全年平均**的数，不是任意时段的数 —— 门限⑥的表述要跟着改。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
