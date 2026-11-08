"""气象因子：Open-Meteo 历史接口（ERA5-Land，约 0.1°），免费、无需密钥（非商业用途）。

说明：ERA5-Land 约 10 km 一格，插值到 30 m 不会产生真实的小尺度差异。
这些因子主要用来解释“火场之间”的整体轻重差别，而不是火场内部的空间差异。
“火灾期间天气”在回测里使用再分析数据，相当于“完美天气预报”情景，论文中需写明。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from . import http

HOURLY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
          "vapour_pressure_deficit", "precipitation", "soil_moisture_0_to_7cm"]


def fetch_points(cfg: dict, lats, lons, start: date, end: date) -> list[pd.DataFrame]:
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in lats),
        "longitude": ",".join(f"{v:.4f}" for v in lons),
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY), "models": "era5_land", "timezone": "Asia/Shanghai",
    }
    r = http.get(cfg["weather"]["archive_url"], params=params)
    r.raise_for_status()
    js = r.json()
    js = js if isinstance(js, list) else [js]
    out = []
    for j in js:
        df = pd.DataFrame(j["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        out.append(df.set_index("time"))
    return out


def summarize(df: pd.DataFrame, ign: date, end: date, ante: list[int]) -> dict:
    d = df.resample("D").agg({"temperature_2m": "max", "relative_humidity_2m": "min",
                              "wind_speed_10m": "max", "vapour_pressure_deficit": "max",
                              "precipitation": "sum", "soil_moisture_0_to_7cm": "mean"})
    fire = d.loc[str(ign):str(end)]
    before = d.loc[:str(ign - timedelta(days=1))]
    out = {
        "fire_tmax": fire["temperature_2m"].max(),
        "fire_rhmin": fire["relative_humidity_2m"].min(),
        "fire_windmax": fire["wind_speed_10m"].max(),
        "fire_vpdmax": fire["vapour_pressure_deficit"].max(),
        "soilm_7d": before["soil_moisture_0_to_7cm"].tail(7).mean(),
        "dry_days_30d": int((before["precipitation"].tail(30) < 1.0).sum()),
    }
    for n in ante:
        out[f"prcp_{n}d"] = before["precipitation"].tail(n).sum()
    return {k: float(v) for k, v in out.items()}


def build_weather(cfg: dict, grid, fire: dict) -> dict:
    ign = date.fromisoformat(str(fire["ignition_date"]))
    end = date.fromisoformat(str(fire["end_date"]))
    ante = cfg["weather"]["antecedent_days"]
    n = cfg["weather"]["grid_points"]
    minx, miny, maxx, maxy = grid.bbox_lonlat
    lons = np.linspace(minx, maxx, n)
    lats = np.linspace(miny, maxy, n)
    LO, LA = np.meshgrid(lons, lats)
    dfs = fetch_points(cfg, LA.ravel(), LO.ravel(), ign - timedelta(days=max(ante)), end)
    pts = pd.DataFrame([summarize(df, ign, end, ante) for df in dfs])
    pts["lon"], pts["lat"] = LO.ravel(), LA.ravel()
    return idw_to_grid(pts, grid)


def idw_to_grid(pts: pd.DataFrame, grid, power: float = 2.0) -> dict:
    """反距离加权插值到网格（只是平滑过渡，不增加信息）。"""
    import xarray as xr
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{grid.epsg}", always_xy=True)
    px, py = tr.transform(pts["lon"].values, pts["lat"].values)
    coords = grid.geobox.coordinates
    ys, xs = coords["y"].values, coords["x"].values
    X, Y = np.meshgrid(xs, ys)
    d = np.sqrt((X[..., None] - px) ** 2 + (Y[..., None] - py) ** 2).clip(min=1.0)
    w = 1.0 / d ** power
    w /= w.sum(-1, keepdims=True)
    out = {}
    for col in pts.columns.drop(["lon", "lat"]):
        v = (w * pts[col].values).sum(-1).astype("float32")
        out[col] = xr.DataArray(v, coords={"y": ys, "x": xs}, dims=("y", "x"))
    return out


def fire_daily(cfg: dict, fire: dict) -> pd.DataFrame:
    """火灾期间（起火前 1 天到结束后 1 天）中心点的逐日天气，用于“燃烧当天天气”。"""
    ign = date.fromisoformat(str(fire["ignition_date"]))
    end = date.fromisoformat(str(fire["end_date"]))
    df = fetch_points(cfg, [float(fire["center_lat"])], [float(fire["center_lon"])],
                      ign - timedelta(days=1), end + timedelta(days=1))[0]
    d = df.resample("D").agg({"temperature_2m": "max", "relative_humidity_2m": "min",
                              "wind_speed_10m": "max", "vapour_pressure_deficit": "max"})
    d.columns = ["tmax", "rhmin", "windmax", "vpdmax"]
    return d
