"""列名 -> token 的解析规则。

设计约束（来自设计文档 §4.2）：本模块不含任何按站点写死的配置。
站点差异只体现在「哪些设备存在」「哪些字段有数据」，两者都从 CSV 自己读出来。

四分（§3.1）:
    a  可控动作  : 频率、启停、冷冻供水温设定（准动作，G3）
    d  外生扰动  : 湿球、站负荷、时间编码、steady
    o  观测      : 各功率、各温度、plr
    z  隐状态    : 网络内部，不在此处

实测依据（reports/audit_yb3_topology_complete.md）:
    - on=0 时温度列恒为 0，on=1 时从不为 0 -> 停机的温度是「无读数」不是「0 度」
    - yb3 tower 0..15 的 consumption 全 NaN，只有 16..19 有 -> 逐塔功率标签不可用
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

CTRL_PERIOD = 900.0  # 秒

FAMILIES = ("plant", "chiller", "tower", "coolpump", "coldpump")
DEVICE_FAMILIES = ("chiller", "tower", "coolpump", "coldpump")

# 设备族的列名前缀。新站点若命名不同，在此追加候选前缀即可，不需改其它代码。
FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "chiller": ("chiller_group/chiller",),
    "tower": ("tower_group/tower",),
    "coolpump": ("cooling_group/cooling_pump",),
    "coldpump": ("colding_group/colding_pump",),
}

# 站级列的候选名 -> 规范名
PLANT_ALIASES: dict[str, tuple[str, ...]] = {
    "wet_bulb": ("wet_bulb_temp", "wet_bulb"),
    "load": ("load",),
    "tower_out": ("tower_out_temp",),
    "steady": ("steady",),
    "agg_chiller_power": ("chiller_consumption",),
    "agg_tower_power": ("tower_consumption",),
    "agg_coolpump_power": ("cool_pump_consumption",),
    "agg_coldpump_power": ("cold_pump_consumption",),
}

TIME_ALIASES = ("time_format", "time", "timestamp", "datetime")

# 每族的规范字段顺序。决定该族 token 的原始特征维 F_type。
# 顺序固定，跨站一致；某站没有的字段整列填 0 且 avail=0。
CANON_FIELDS: dict[str, tuple[str, ...]] = {
    "plant": ("wet_bulb", "load", "tower_out", "steady",
              "hour_sin", "hour_cos", "dow_sin", "dow_cos", "dt_ratio"),
    "chiller": ("on", "cold_out_temp", "plr", "cold_back_temp",
                "cool_back_temp", "cool_out_temp", "consumption"),
    "tower": ("on", "frequency", "consumption"),
    "coolpump": ("on", "frequency", "consumption"),
    "coldpump": ("on", "frequency", "consumption"),
}

# 四分角色。(family, field) -> "a" | "d" | "o"
FIELD_ROLE: dict[tuple[str, str], str] = {
    ("plant", "wet_bulb"): "d",
    ("plant", "load"): "d",
    ("plant", "steady"): "d",
    ("plant", "hour_sin"): "d",
    ("plant", "hour_cos"): "d",
    ("plant", "dow_sin"): "d",
    ("plant", "dow_cos"): "d",
    ("plant", "dt_ratio"): "d",
    ("plant", "tower_out"): "o",

    ("chiller", "on"): "a",
    # G3：数据里只有实测值没有设定值，暂按「准动作」处理，扰动幅度须限制在实测分布内
    ("chiller", "cold_out_temp"): "a",
    ("chiller", "plr"): "o",
    ("chiller", "cold_back_temp"): "o",
    ("chiller", "cool_back_temp"): "o",
    ("chiller", "cool_out_temp"): "o",
    ("chiller", "consumption"): "o",
}
for _fam in ("tower", "coolpump", "coldpump"):
    FIELD_ROLE[(_fam, "on")] = "a"
    FIELD_ROLE[(_fam, "frequency")] = "a"
    FIELD_ROLE[(_fam, "consumption")] = "o"

# 停机时读数无效的字段（实测：on=0 时恒为 0，on=1 时从不为 0）。
# 功率不在此列 —— 停机功耗确实是 0，那是真值。
OFF_INVALID_FIELDS = frozenset({
    "cold_out_temp", "cold_back_temp", "cool_back_temp", "cool_out_temp",
})

# 量程规则，仅在 avail=1 的位置上检查/裁剪
RANGE_RULES: dict[str, tuple[float, float]] = {
    "frequency": (0.0, 60.0),
    "plr": (0.0, 1.5),
    "on": (0.0, 1.0),
    "cold_out_temp": (0.0, 30.0),
    "cold_back_temp": (0.0, 35.0),
    "cool_back_temp": (5.0, 50.0),
    "cool_out_temp": (5.0, 55.0),
    "tower_out": (5.0, 50.0),
    "wet_bulb": (-20.0, 40.0),
    "consumption": (0.0, 1e5),
}


def role_index(family: str) -> dict[str, list[int]]:
    """该族规范字段里，各角色占哪些下标。"""
    out: dict[str, list[int]] = {"a": [], "d": [], "o": []}
    for i, f in enumerate(CANON_FIELDS[family]):
        out[FIELD_ROLE[(family, f)]].append(i)
    return out


@dataclass(frozen=True)
class SiteSchema:
    """一个站点解析出来的结构。纯数据，不含权重。"""

    site: str
    n_dev: dict[str, int]                      # family -> 设备台数
    col_of: dict[tuple[str, int, str], str]    # (family, dev, field) -> CSV 列名
    plant_col: dict[str, str]                  # 规范名 -> CSV 列名
    time_col: str

    @property
    def n_tokens(self) -> int:
        return 1 + sum(self.n_dev[f] for f in DEVICE_FAMILIES)

    @property
    def token_index(self) -> list[tuple[str, int]]:
        """token 顺序：plant, chiller*, tower*, coolpump*, coldpump*。"""
        idx: list[tuple[str, int]] = [("plant", 0)]
        for fam in DEVICE_FAMILIES:
            idx.extend((fam, d) for d in range(self.n_dev[fam]))
        return idx

    @property
    def type_id(self) -> np.ndarray:
        """每个 token 的族编号，[N]，用于 Emb_type 与 attn_bias。"""
        order = {f: i for i, f in enumerate(FAMILIES)}
        return np.array([order[f] for f, _ in self.token_index], dtype=np.int64)

    def describe(self) -> str:
        parts = " + ".join(f"{f} {self.n_dev[f]}" for f in DEVICE_FAMILIES)
        return f"{self.site}: N={self.n_tokens} = plant 1 + {parts}"


def _scan_family(cols: list[str], family: str) -> tuple[list[int], dict[tuple[int, str], str]]:
    ids: set[int] = set()
    found: dict[tuple[int, str], str] = {}
    for prefix in FAMILY_PREFIXES[family]:
        pat = re.compile(rf"^{re.escape(prefix)}/(\d+)/(.+)$")
        for c in cols:
            m = pat.match(c)
            if m:
                dev, fld = int(m.group(1)), m.group(2)
                ids.add(dev)
                found[(dev, fld)] = c
    return sorted(ids), found


def parse_site(df: pd.DataFrame, site: str) -> SiteSchema:
    """从 DataFrame 的列名反推站点结构。不读取任何外部配置。"""
    cols = df.columns.tolist()

    time_col = next((c for c in TIME_ALIASES if c in cols), None)
    if time_col is None:
        raise ValueError(f"{site}: 找不到时间列，候选 {TIME_ALIASES}")

    n_dev: dict[str, int] = {}
    col_of: dict[tuple[str, int, str], str] = {}
    for fam in DEVICE_FAMILIES:
        ids, found = _scan_family(cols, fam)
        # 设备 id 必须是 0..n-1 连续，否则 token 顺序与 id 对不上
        if ids and ids != list(range(len(ids))):
            raise ValueError(f"{site}/{fam}: 设备编号不连续 {ids}")
        n_dev[fam] = len(ids)
        for dev in ids:
            for fld in CANON_FIELDS[fam]:
                if (dev, fld) in found:
                    col_of[(fam, dev, fld)] = found[(dev, fld)]

    plant_col = {}
    for canon, aliases in PLANT_ALIASES.items():
        hit = next((a for a in aliases if a in cols), None)
        if hit is not None:
            plant_col[canon] = hit

    return SiteSchema(site=site, n_dev=n_dev, col_of=col_of,
                      plant_col=plant_col, time_col=time_col)


def _time_features(ts: pd.Series) -> dict[str, np.ndarray]:
    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
    dow = ts.dt.dayofweek.to_numpy().astype(float)
    # pandas 3 的 to_numpy() 默认返回只读视图（copy-on-write），必须显式拷贝
    dt = np.array(ts.diff().dt.total_seconds().to_numpy(), dtype=float, copy=True)
    dt[0] = CTRL_PERIOD
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        # 段内理论恒为 900s；跨断点会很大，但那些行会被段切分排除
        "dt_ratio": np.log(np.clip(dt, 1.0, None) / CTRL_PERIOD),
    }


def build_arrays(df: pd.DataFrame, sch: SiteSchema, *, clip_range: bool = True
                 ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """把 DataFrame 铺成 token 张量。

    返回:
        x     [T, N, F_max]  数值，无效位置置 0
        avail [T, N, F_max]  1=有效, 0=缺测/停机无读数/该族无此字段
        extra 站级派生量（逐台功率求和、聚合功率等），供损失与描述符使用
    """
    T = len(df)
    N = sch.n_tokens
    F_max = max(len(v) for v in CANON_FIELDS.values())

    x = np.zeros((T, N, F_max), dtype=np.float32)
    avail = np.zeros((T, N, F_max), dtype=np.float32)

    tfeat = _time_features(pd.to_datetime(df[sch.time_col]))

    for n, (fam, dev) in enumerate(sch.token_index):
        fields = CANON_FIELDS[fam]

        # 该设备的 on 序列，用于停机掩码
        on = None
        if fam in DEVICE_FAMILIES and (fam, dev, "on") in sch.col_of:
            on = pd.to_numeric(df[sch.col_of[(fam, dev, "on")]], errors="coerce").to_numpy()

        for k, fld in enumerate(fields):
            if fam == "plant":
                if fld in tfeat:
                    v = tfeat[fld]
                    m = np.isfinite(v)
                elif fld in sch.plant_col:
                    v = pd.to_numeric(df[sch.plant_col[fld]], errors="coerce").to_numpy()
                    m = np.isfinite(v)
                else:
                    continue  # 该站没有这一列，整列 avail=0
            else:
                key = (fam, dev, fld)
                if key not in sch.col_of:
                    continue
                v = pd.to_numeric(df[sch.col_of[key]], errors="coerce").to_numpy()
                m = np.isfinite(v)
                # 停机 -> 温度无读数
                if fld in OFF_INVALID_FIELDS and on is not None:
                    m = m & (on == 1)

            v = np.asarray(v, dtype=np.float64)
            if clip_range and fld in RANGE_RULES:
                lo, hi = RANGE_RULES[fld]
                m = m & (v >= lo) & (v <= hi)

            x[:, n, k] = np.where(m, np.nan_to_num(v), 0.0)
            avail[:, n, k] = m.astype(np.float32)

    extra = _plant_extras(df, sch)
    return x, avail, extra


def _plant_extras(df: pd.DataFrame, sch: SiteSchema) -> dict[str, np.ndarray]:
    """站级派生量：逐台功率求和 与 聚合列，以及主口径总功率。

    口径规则（G2 裁决）：
        1. 「从未开机」的设备（on 恒为 0）视为不存在，不计入台数 —— yb3 的 coldpump/7
           属于此类，它的 consumption 全 NaN 但物理功率确实是 0。
        2. 剩余设备若全部有逐台 consumption，采信逐台求和（更可信，能对上分项）；
           否则采信站级聚合列。
        3. 逐台监督（损失 k=5）按**设备**开关，不按族开关 —— 有标签的那几台照样监督。

    实测（yb3）：tower 仅 4/20 台有逐台功率 -> 塔功率走聚合，且无逐塔监督；
                chiller/coolpump 7/7 全有 -> 逐台求和 + 逐台监督。
    """
    out: dict[str, np.ndarray] = {}
    T = len(df)
    total = np.zeros(T, dtype=np.float64)
    total_ok = np.ones(T, dtype=bool)

    for fam in DEVICE_FAMILIES:
        n_dev = sch.n_dev[fam]
        has_label = np.zeros(n_dev, dtype=bool)   # 该台有逐台功率标签
        is_phantom = np.zeros(n_dev, dtype=bool)  # 该台从未开机
        per = np.zeros((T, n_dev))

        for d in range(n_dev):
            key = (fam, d, "consumption")
            if key in sch.col_of:
                v = pd.to_numeric(df[sch.col_of[key]], errors="coerce").to_numpy()
                has_label[d] = bool(np.isfinite(v).any())
                per[:, d] = np.nan_to_num(v)
            on_key = (fam, d, "on")
            if on_key in sch.col_of:
                on = pd.to_numeric(df[sch.col_of[on_key]], errors="coerce").to_numpy()
                is_phantom[d] = not bool(np.nansum(on) > 0)

        real = ~is_phantom
        per_full = bool(n_dev > 0 and (has_label | is_phantom).all())
        per_sum = per[:, real].sum(axis=1) if real.any() else np.zeros(T)

        agg_key = f"agg_{fam}_power"
        agg = (pd.to_numeric(df[sch.plant_col[agg_key]], errors="coerce").to_numpy()
               if agg_key in sch.plant_col else np.full(T, np.nan))

        if per_full:
            val, src = per_sum, "per_device"
        elif np.isfinite(agg).any():
            val, src = agg, "aggregate"
        else:
            val, src = np.zeros(T), "missing"

        out[f"power_{fam}"] = np.nan_to_num(val).astype(np.float32)
        out[f"power_{fam}_ok"] = np.isfinite(val).astype(np.float32)
        out[f"power_{fam}_src"] = src
        out[f"power_{fam}_n_label"] = int(has_label.sum())
        out[f"power_{fam}_n_phantom"] = int(is_phantom.sum())
        # 逐台监督掩码：该台有标签且非幽灵 -> 进 k=5 损失
        out[f"per_device_mask_{fam}"] = (has_label & real).astype(np.float32)
        out[f"phantom_{fam}"] = is_phantom.astype(np.float32)

        total += np.nan_to_num(val)
        total_ok &= np.isfinite(val)

    out["P_plant"] = total.astype(np.float32)
    out["P_plant_ok"] = total_ok.astype(np.float32)
    return out


def split_segments(df: pd.DataFrame, sch: SiteSchema, *, min_len: int
                   ) -> list[tuple[int, int]]:
    """按 900s 断点切连续段，只保留长度 >= min_len 的段。返回 [start, end) 列表。"""
    ts = pd.to_datetime(df[sch.time_col])
    dt = ts.diff().dt.total_seconds().to_numpy()
    is_break = np.ones(len(df), dtype=bool)
    is_break[1:] = dt[1:] != CTRL_PERIOD
    starts = np.flatnonzero(is_break)
    ends = np.append(starts[1:], len(df))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]
