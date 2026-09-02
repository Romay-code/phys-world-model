"""14 个「站点」里有几个是**互相独立**的冷站？

`multisite.DUP_FILES` 只去掉了同名重复（yb3 家族 4 个文件、zx 家族 2 个），
得到 14 个「独立站点」。但 `pa2_中温/低温`、`yb_中温/低温`、`yc_中温/低温`
这三对，名字就写着是**同一冷站的中温/低温两个温区回路**。

若属实，后果是实打实的：

  · 留出 `yb_低温` 而把 `yb_中温` 留在训练里，不是跨站零样本 ——
    同楼、同天气、同负荷排程，很可能还共用冷却塔。这个「零样本」偏乐观。
  · 标度曲线的横轴「站点数 ∈ {1,2,4,8,14}」其实只有 11 个独立冷站，
    曲线的解释要跟着改。

本工具用三项**可证伪**的证据判定，不靠名字：

  1. 行数与时间轴是否逐位一致（同一套采集时钟）
  2. 站级气象量 `wet_bulb` 是否逐点一致（同一地点、同一时刻）
  3. 设备台数是否不同（若连台数都一样，可能干脆是重复文件而非两个回路）

实测结论：三对全部判定为同一冷站。`yb` 两回路的湿球**逐点相同**（中位差
0.000 K，同一个传感器）；`yc` 中位差 0.24 K、`pa2` 0.43 K，是同址的两个
传感器。台数各不相同（5/14 vs 4/11 等），确实是两套设备的两个回路。

**hx 不受影响** —— 它没有兄弟回路，P5-A 的 hx 零样本 R²=0.9335 仍是干净的跨站数。
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402
from physwm.data.dataset import load_site  # noqa: E402

DUP = {"yb3_topology_clean_v2", "yb3_topology_complete", "yb3test", "zxtest"}


def main() -> int:
    sites = {}
    for f in sorted((ROOT / "data").glob("*.csv")):
        if f.stem in DUP:
            continue
        sd = load_site(f, f.stem)
        df = pd.read_csv(f, low_memory=False)
        ts = pd.to_datetime(df[sd.sch.time_col], errors="coerce")
        k = S.CANON_FIELDS["plant"].index("wet_bulb")
        sites[f.stem] = {"T": sd.T, "ts": ts.to_numpy(),
                         "wb": sd.x[:, 0, k].astype(np.float64),
                         "wb_av": sd.avail[:, 0, k].astype(bool),
                         "n_dev": dict(sd.sch.n_dev)}

    print(f"{'站点对':<34}{'同行数':>7}{'同时间轴':>9}{'wet_bulb 相关':>13}"
          f"{'湿球中位差K':>11}   台数(冷机/塔)")
    print("-" * 104)
    hits = []
    for a, b in itertools.combinations(sites, 2):
        A, B = sites[a], sites[b]
        same_T = A["T"] == B["T"]
        if not same_T:
            continue                                  # 行数不同就不可能是同一时钟
        same_ts = bool((A["ts"] == B["ts"]).all())
        m = A["wb_av"] & B["wb_av"]
        if m.sum() > 100:
            corr = float(np.corrcoef(A["wb"][m], B["wb"][m])[0, 1])
            med = float(np.median(np.abs(A["wb"][m] - B["wb"][m])))
        else:
            corr, med = float("nan"), float("nan")
        # 判据不能用「逐位相同」：实测 yc 两回路的湿球中位差 0.24 K、
        # pa2 是 0.43 K —— 同一地点的两个传感器本来就不会逐位相同，
        # 而 yb 两回路是 0.000（同一个传感器）。**同一套时间轴 + 湿球相关 > 0.99**
        # 才是「同址同时」的可证伪判据，逐位相同是它的特例。
        same_site = same_ts and np.isfinite(corr) and corr > 0.99
        dev = (f"{A['n_dev']['chiller']}/{A['n_dev']['tower']} vs "
               f"{B['n_dev']['chiller']}/{B['n_dev']['tower']}")
        print(f"{a + ' | ' + b:<34}{'是' if same_T else '否':>7}"
              f"{'是' if same_ts else '否':>9}{corr:>13.4f}"
              f"{med:>11.3f}   {dev}")
        if same_site:
            hits.append((a, b))

    print()
    if hits:
        print("**判定为同一冷站的不同回路**（时间轴逐位一致 + 湿球相关 > 0.99）：")
        for a, b in hits:
            print(f"  · {a}  ==  {b}")
        merged = set()
        for a, b in hits:
            merged |= {a, b}
        n_indep = len(sites) - len(hits)
        print()
        print(f"文件数（去同名重复后）{len(sites)}  ->  **独立冷站 {n_indep} 个**")
        print("留出设计必须把同一冷站的两个回路**一起**留出，否则「零样本」偏乐观。")
    else:
        print("未发现同一冷站的成对回路。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
