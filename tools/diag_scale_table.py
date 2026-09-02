"""把逐站量纲 `infer_site_scales` 直接打出来，并与该站实测的典型值对照。

多站零样本的失败若是「标定问题」，源头只可能在这张表：解码器的物理量
全部由这些 buffer 承载量纲，它们错了不会报错、不会违例，只会让数偏掉。

对照列 `实测每台冷机功率` 用与 `infer_w_scale` 同样的定义直接从数据算，
两者理应相等 —— 不等就说明推断路径上有分支走岔了。
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


def main() -> int:
    spec = WindowSpec()
    hdr = ("站点", "w_scale", "实测每台", "p_tower", "p_coolp", "p_coldp",
           "q_scale", "k_mcp", "mcp_sc", "dt_ev", "P_plant中位")
    print(f"{hdr[0]:<18}" + "".join(f"{h:>11}" for h in hdr[1:]))
    print("-" * 130)
    for f in sorted((ROOT / "data").glob("*.csv")):
        if f.stem in DUP:
            continue
        b = build_site_bundle(f, f.stem, spec, out_dir=None)
        sd = b["data"]
        rows = covered_rows(b["splits"]["train"], spec, sd.T)
        sc = infer_site_scales(sd, rows)

        # 与 infer_w_scale 同定义的独立复算
        k_on = S.CANON_FIELDS["chiller"].index("on")
        ch = [i for i, (ff, _) in enumerate(sd.sch.token_index) if ff == "chiller"]
        n_on = sd.x[rows][:, ch, k_on].sum(-1)
        w = sd.extra["power_chiller"][rows]
        m = (n_on > 0) & (w > 0)
        ref = float(np.median(w[m] / n_on[m])) if m.sum() > 32 else float("nan")
        pp = float(np.median(sd.extra["P_plant"][rows]))

        print(f"{f.stem:<18}{sc['w_scale']:>11.1f}{ref:>11.1f}"
              f"{sc['p_scale_tower']:>11.1f}{sc['p_scale_coolpump']:>11.1f}"
              f"{sc['p_scale_coldpump']:>11.1f}{sc['q_scale']:>11.1f}"
              f"{sc['k_mcp']:>11.2f}{sc['mcp_scale']:>11.1f}"
              f"{sc['dt_evap_scale']:>11.2f}{pp:>11.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
