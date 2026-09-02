"""全站 × 全通道的输入可用率矩阵。

`probe_avail_shift.py` 只测了 `chiller.consumption` 一个通道，因为最初的假设
只点了它。但那个假设已被 hx 的对照否掉（hx 同为零样本、掩掉该通道只从
0.938 掉到 0.851），所以必须把「pc3 到底缺哪些通道」整张摊开，
而不是继续逐个猜。

逐 (族, 字段) 报该站所有该族设备上的 avail 均值（在训练覆盖行上统计）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402
from physwm.data.dataset import WindowSpec, build_site_bundle, covered_rows  # noqa: E402

DUP = {"yb3_topology_clean_v2", "yb3_topology_complete", "yb3test", "zxtest"}


def main() -> int:
    spec = WindowSpec()
    cols = []
    for fam in ("plant",) + tuple(S.DEVICE_FAMILIES):
        for fld in S.CANON_FIELDS[fam]:
            cols.append((fam, fld))

    rows_out = {}
    for f in sorted((ROOT / "data").glob("*.csv")):
        if f.stem in DUP:
            continue
        b = build_site_bundle(f, f.stem, spec, out_dir=None)
        sd = b["data"]
        rows = covered_rows(b["splits"]["train"], spec, sd.T)
        vals = {}
        for fam, fld in cols:
            idx = [i for i, (ff, _) in enumerate(sd.sch.token_index) if ff == fam]
            k = S.CANON_FIELDS[fam].index(fld)
            vals[(fam, fld)] = (float(sd.avail[rows][:, idx, k].mean())
                                if idx else float("nan"))
        rows_out[f.stem] = vals

    names = list(rows_out)
    for fam in ("plant",) + tuple(S.DEVICE_FAMILIES):
        flds = [c for c in cols if c[0] == fam]
        print(f"\n=== {fam} ===")
        print(f"{'站点':<18}" + "".join(f"{fld:>15}" for _, fld in flds))
        print("-" * (18 + 15 * len(flds)))
        for n in names:
            print(f"{n:<18}" + "".join(f"{rows_out[n][c]:>15.3f}" for c in flds))

    # 差异定位：pc3 与 12 个训练站中位的差
    tgt = "pc3"
    tr = [n for n in names if n not in (tgt, "hx")]
    print(f"\n\n=== {tgt} 与 12 训练站中位可用率之差（只列 |差| > 0.2）===")
    print(f"{'通道':<30}{'pc3':>10}{'训练站中位':>12}{'差':>10}")
    print("-" * 62)
    hits = []
    for c in cols:
        med = float(np.nanmedian([rows_out[n][c] for n in tr]))
        d = rows_out[tgt][c] - med
        if np.isfinite(d) and abs(d) > 0.2:
            hits.append((d, c, rows_out[tgt][c], med))
    for d, c, v, med in sorted(hits):
        print(f"{c[0] + '.' + c[1]:<30}{v:>10.3f}{med:>12.3f}{d:>10.3f}")
    if not hits:
        print("（无）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
