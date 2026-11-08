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
    for col, name in [("tmax", "bd_tmax"), ("rhmin", "bd_rhmin"), ("windmax", "bd_windmax"),
                      ("vpdmax", "bd_vpdmax")]:
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
