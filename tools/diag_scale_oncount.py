"""诊断：逐族功率量纲 `p_scale_*` 的分母用错了台数。

`scales.infer_w_scale`（冷机）除以**开机台数**的中位：

    median(power_chiller / n_on)          -> 每台开机冷机的典型功率

而 `scales.infer_scales`（塔/冷却泵/冷冻泵）除以**装机台数**：

    median(power_fam[p>0]) / n_dev        -> 每台**装机**设备的典型功率

解码器按 `w = on · (fr/50)³ · p_scale` 逐台求和，求和只跑在**开机**的设备上。
于是当一个站装机多、常开少时，`p_scale` 被系统性低估 `n_dev / n_on` 倍，
该族功率随之整体偏小 —— 不报错、不违例，只是数偏小。

本表给出每站每族的 `n_dev`、开机台数中位、二者之比。比值越大，偏得越狠。

    python tools/diag_scale_oncount.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402
from physwm.data.dataset import WindowSpec, build_site_bundle, covered_rows  # noqa: E402
from physwm.data.scales import infer_site_scales  # noqa: E402

DUP = {"yb3_topology_clean_v2", "yb3_topology_complete", "yb3test", "zxtest"}
FAMS = ("tower", "coolpump", "coldpump")


def main() -> int:
    spec = WindowSpec()
    print(f"{'站点':<18}" + "".join(f"{f+' 装机/常开(比)':>22}" for f in FAMS))
    print("-" * 90)
    worst = []
    for f in sorted((ROOT / "data").glob("*.csv")):
        if f.stem in DUP:
            continue
        b = build_site_bundle(f, f.stem, spec, out_dir=None)
        sd = b["data"]
        rows = covered_rows(b["splits"]["train"], spec, sd.T)
        cells, ratios = [], []
        for fam in FAMS:
            idx = [i for i, (ff, _) in enumerate(sd.sch.token_index) if ff == fam]
            n_dev = len(idx)
            if not n_dev:
                cells.append(f"{'-':>22}"); continue
            k_on = S.CANON_FIELDS[fam].index("on")
            n_on = sd.x[rows][:, idx, k_on].sum(-1)
            med = float(np.median(n_on[n_on > 0])) if (n_on > 0).any() else 0.0
            r = n_dev / med if med > 0 else float("inf")
            ratios.append(r)
            cells.append(f"{n_dev:>10d}/{med:<5.1f}({r:>4.1f})")
        print(f"{f.stem:<18}" + "".join(cells))
        worst.append((max(ratios) if ratios else 0.0, f.stem))
    print()
    print("按最大偏差倍数排序：")
    for r, n in sorted(worst, reverse=True):
        print(f"  {n:<20}{r:>6.1f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
