"""火中观测因子（来自 VIIRS 火点）：每个像元离最近火点多远、附近火点数量和燃烧强度（FRP）、
大约哪天被烧到，以及那天的天气。

这些信息在火灾发生时（或刚结束时）就能拿到，可以用于“火灾发生后、尚无火后影像时”的快速预测，
但不属于“火前”信息。论文中应与只用火前因子的结果分开报告。
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import fftconvolve


def _disk(radius_px: int) -> np.ndarray:
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    return (x * x + y * y <= radius_px * radius_px).astype("float32")


def hotspot_layers(hs: pd.DataFrame, shape: tuple, res_m: float, ignition: date,
                   radius_m: float = 750) -> dict[str, np.ndarray]:
    """hs 需要列：row, col, acq_date, frp。"""
    ok = (hs["row"] >= 0) & (hs["row"] < shape[0]) & (hs["col"] >= 0) & (hs["col"] < shape[1])
    hs = hs[ok]
    out = {}
    if hs.empty:
        for k in ("hs_dist", "hs_count_750", "hs_frp_max_750", "hs_frp_nearest", "burn_day"):
            out[k] = np.full(shape, np.nan, "float32")
        return out
    day = np.array([(d - ignition).days for d in hs["acq_date"]], dtype="float32")
    cnt = np.zeros(shape, "float32")
    frp = np.zeros(shape, "float32")
    first = np.full(shape, np.inf, "float32")
    for r, c, d, f in zip(hs["row"].values, hs["col"].values, day, hs["frp"].fillna(0).values):
        cnt[r, c] += 1
        frp[r, c] = max(frp[r, c], f)
        first[r, c] = min(first[r, c], d)
    hot = cnt > 0
    dist, (ir, ic) = ndimage.distance_transform_edt(~hot, return_indices=True)
    k = _disk(max(1, int(round(radius_m / res_m))))
    out["hs_dist"] = (dist * res_m).astype("float32")
    out["hs_count_750"] = np.rint(fftconvolve(cnt, k, mode="same")).astype("float32")
    out["hs_frp_max_750"] = ndimage.maximum_filter(frp, footprint=k.astype(bool)).astype("float32")
    out["hs_frp_nearest"] = frp[ir, ic].astype("float32")
    out["burn_day"] = first[ir, ic].astype("float32")
    return out


def burn_day_weather(burn_day: np.ndarray, daily: pd.DataFrame, ignition: date) -> dict[str, np.ndarray]:
    """把每个像元“被烧到那天”的天气填进去。daily 以日期为索引，列为 tmax/rhmin/windmax/vpdmax。"""
    out = {}
    idx = {(pd.Timestamp(d).date() - ignition).days: i for i, d in enumerate(daily.index)}
    bd = np.nan_to_num(burn_day, nan=0).astype(int)
    lookup = np.vectorize(lambda v: idx.get(v, idx.get(min(idx, key=lambda k: abs(k - v)))))
    rows = lookup(bd)
    pairs = [("tmax", "bd_tmax"), ("rhmin", "bd_rhmin"), ("windmax", "bd_windmax"), ("vpdmax", "bd_vpdmax")]
    if "winddir" in daily:
        pairs.append(("winddir", "bd_winddir"))
    for col, name in pairs:
        vals = daily[col].to_numpy(dtype="float32")
        arr = vals[rows]
        arr[~np.isfinite(burn_day)] = np.nan
        out[name] = arr.astype("float32")
    return out


def context_layers(layers: dict[str, np.ndarray], res_m: float, radius_m: float = 150) -> dict[str, np.ndarray]:
    """火前植被的邻域特征（约 300 m 窗口）：周围植被的平均状态和斑块异质性。"""
    size = max(3, int(2 * radius_m / res_m) | 1)

    def nanmean(a):
        v = np.nan_to_num(a, nan=0.0)
        w = np.isfinite(a).astype("float32")
        s = ndimage.uniform_filter(v, size) / np.maximum(ndimage.uniform_filter(w, size), 1e-6)
        s[ndimage.uniform_filter(w, size) < 0.2] = np.nan
        return s

    m = nanmean(layers["nbr_pre"])
    ndvi = layers["ndvi_pre"]
    sd = np.sqrt(np.maximum(nanmean(ndvi ** 2) - nanmean(ndvi) ** 2, 0))
    return {"nbr_pre_mean300": m.astype("float32"), "ndvi_pre_sd300": sd.astype("float32")}


def _ang_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两个方位角（度）的夹角，0–180。"""
    d = np.abs((a - b + 180) % 360 - 180)
    return d.astype("float32")


def fire_behavior_layers(hs: pd.DataFrame, shape: tuple, res_m: float, ignition: date,
                         aspect_deg: np.ndarray, slope_deg: np.ndarray,
                         bd_winddir: np.ndarray | None, bd_windmax: np.ndarray | None,
                         search_m: float = 3000) -> dict[str, np.ndarray]:
    """火行为因子：让气象在火场内部产生空间差异。

    - aspect_wind_angle：坡向与“风去的方向”的夹角。0 表示风正往这面坡上吹（迎风坡，顺风上坡），180 表示背风坡。
    - wind_slope_index：风速 × cos(夹角) × tan(坡度)，顺风上坡为正、背风下坡为负。
    - spread_dir：火到达这个像元的方向（从更早的火点指向这里的方位角）。
    - spread_aspect_angle / spread_wind_angle：蔓延方向与坡向（上坡方向）、风向的夹角。
    - spread_speed：相邻两次观测之间的蔓延速度（m/h）。

    hs 需要列 row, col, acq_date, acq_time(HHMM), frp。蔓延方向对每个火点找它之前（≥1 小时）最近的火点。
    """
    out = {}
    nan = lambda: np.full(shape, np.nan, "float32")
    upslope = (aspect_deg + 180) % 360               # 坡向是坡面朝向（下坡方向），上坡方向加 180
    # 1) 坡向-风向夹角、风坡指数（逐像元，用燃烧当天风向）
    if bd_winddir is not None and bd_windmax is not None:
        wind_to = (bd_winddir + 180) % 360           # 气象风向是风来的方向，火被推向相反方向
        ang = _ang_diff(upslope, wind_to)
        out["aspect_wind_angle"] = ang
        out["wind_slope_index"] = (bd_windmax * np.cos(np.radians(ang)) * np.tan(np.radians(slope_deg))).astype("float32")
    else:
        out["aspect_wind_angle"], out["wind_slope_index"] = nan(), nan()
    # 2) 蔓延方向与速度：按火点时序
    ok = (hs["row"] >= 0) & (hs["row"] < shape[0]) & (hs["col"] >= 0) & (hs["col"] < shape[1])
    h = hs[ok].copy()
    if len(h) < 2:
        for k in ("spread_dir", "spread_aspect_angle", "spread_wind_angle", "spread_speed"):
            out[k] = nan()
        return out
    t = h["acq_time"].fillna(0).astype(int)
    hours = np.array([(d - ignition).days * 24 for d in h["acq_date"]], "float64") + (t // 100) + (t % 100) / 60.0
    r, c = h["row"].values.astype(float), h["col"].values.astype(float)
    n = len(h)
    sdir = np.full(n, np.nan); sspd = np.full(n, np.nan)
    search_px = search_m / res_m
    for i in range(n):
        prev = np.where(hours <= hours[i] - 1.0)[0]
        if len(prev) == 0:
            continue
        d = np.hypot(r[prev] - r[i], c[prev] - c[i])
        j = prev[np.argmin(d)]
        if d.min() > search_px or d.min() == 0:
            continue
        dr, dc = r[i] - r[j], c[i] - c[j]          # 行向下为南
        sdir[i] = (np.degrees(np.arctan2(dc, -dr)) + 360) % 360
        sspd[i] = d.min() * res_m / max(hours[i] - hours[j], 1.0)
    # 把每个像元归到最近的火点，取该火点的蔓延方向/速度
    cnt = np.zeros(shape, bool)
    cnt[h["row"].values, h["col"].values] = True
    _, (ir, ic) = ndimage.distance_transform_edt(~cnt, return_indices=True)
    g_dir = np.full(shape, np.nan, "float32"); g_spd = np.full(shape, np.nan, "float32")
    # 同一像元多个火点：取有方向的最后一个
    for k in range(n):
        if np.isfinite(sdir[k]):
            g_dir[int(r[k]), int(c[k])] = sdir[k]; g_spd[int(r[k]), int(c[k])] = sspd[k]
    out["spread_dir"] = g_dir[ir, ic]
    out["spread_speed"] = g_spd[ir, ic]
    out["spread_aspect_angle"] = np.where(np.isfinite(out["spread_dir"]), _ang_diff(out["spread_dir"], upslope), np.nan).astype("float32")
    if bd_winddir is not None:
        wind_to = (bd_winddir + 180) % 360
        out["spread_wind_angle"] = np.where(np.isfinite(out["spread_dir"]), _ang_diff(out["spread_dir"], wind_to), np.nan).astype("float32")
    else:
        out["spread_wind_angle"] = nan()
    return out
