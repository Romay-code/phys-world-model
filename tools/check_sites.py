"""P5 前置：验证数据底座在全部站点上能否跑通。

P0–P4 全部工作只用了 yb3 一站，`build_site_bundle` / `compute_descriptors`
从未在其余站点上执行过。P5 的所有工作都压在跨站底座上，所以先把
「哪些站能加载、能切段、能算描述符」变成实测表，而不是从体检报告推断。

    python tools/check_sites.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data.dataset import WindowSpec, build_site_bundle  # noqa: E402
from physwm.data.descriptors import compute_descriptors  # noqa: E402

# 18 个 CSV 去重后 14 个独立站点/回路：yb3 家族 4 个文件、zx 家族 2 个
DUP = {"yb3_topology_clean_v2": "yb3", "yb3_topology_complete": "yb3",
       "yb3test": "yb3", "zxtest": "zx"}


def probe(f: Path, spec: WindowSpec) -> dict:
    out = {"site": f.stem, "indep": "-" if f.stem in DUP else "Y",
           "load": "x", "N": "", "F": "", "win": "", "desc": "", "note": ""}
    try:
        b = build_site_bundle(f, f.stem, spec, out_dir=ROOT / "artifacts" / "_probe")
    except Exception as e:
        out["note"] = f"{type(e).__name__}: {str(e)[:70]}"
        return out
    sd, sp = b["data"], b["splits"]
    out["load"] = "Y"
    out["N"], out["F"] = sd.N, sd.F
    n = {k: len(v) for k, v in sp.items()}
    out["win"] = "%d/%d/%d" % (n["train"], n["val"], n["test"])
    if min(n.values()) == 0:
        out["note"] = "**某个 split 为空**"
    try:
        db = compute_descriptors(sd.x, sd.avail, sd.sch, b["train_rows"])
        out["desc"] = "%dd" % db.desc.shape[1]
    except Exception as e:
        out["desc"] = "x"
        out["note"] = (out["note"] + " 描述符: %s: %s" % (type(e).__name__, str(e)[:50])).strip()
    return out


def main() -> int:
    spec = WindowSpec()
    rows = [probe(f, spec) for f in sorted((ROOT / "data").glob("*.csv"))]
    print("%-24s %-5s %-5s %4s %3s %22s %6s  %s"
          % ("站点", "独立", "加载", "N", "F", "train/val/test", "desc", "备注"))
    print("-" * 104)
    for r in rows:
        print("%-24s %-5s %-5s %4s %3s %22s %6s  %s"
              % (r["site"], r["indep"], r["load"], r["N"], r["F"],
                 r["win"], r["desc"], r["note"]))
    good = [r for r in rows if r["load"] == "Y" and r["desc"].endswith("d") and not r["note"]]
    indep_good = [r for r in good if r["indep"] == "Y"]
    print()
    print("完全跑通 %d / %d 个文件；其中独立站点 %d 个"
          % (len(good), len(rows), len(indep_good)))
    bad = [r for r in rows if r not in good]
    if bad:
        print("有问题的：")
        for r in bad:
            print("  - %-24s %s" % (r["site"], r["note"] or "(desc 失败)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
