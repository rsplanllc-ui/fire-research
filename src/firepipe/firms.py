"""NASA FIRMS 火点（VIIRS 375 m）：自动发现火灾事件、确定起止日期和中心坐标。

需要免费的 MAP_KEY（https://firms.modaps.eosdis.nasa.gov/api/map_key/），
放在环境变量 FIRMS_MAP_KEY（GitHub Secrets）里。
接口限制：每次最多 5 天；每 10 分钟 5000 次请求。
"""
from __future__ import annotations

import io
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from . import http

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{area}/{days}/{date}"
SOURCES = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"]


def fetch(bbox, start: date, end: date, sources=SOURCES, key: str | None = None) -> pd.DataFrame:
    key = (key or os.environ["FIRMS_MAP_KEY"]).strip()
    area = ",".join(f"{v:.4f}" for v in bbox)          # 西,南,东,北
    parts = []
    for src in sources:
        d = start
        while d <= end:
            n = min(5, (end - d).days + 1)
            r = _get_quota_aware(API.format(key=key, src=src, area=area, days=n, date=d.isoformat()))
            if r.text.strip() and not r.text.lower().startswith("invalid"):
                df = pd.read_csv(io.StringIO(r.text))
                if len(df):
                    df["source"] = src
                    parts.append(df)
            d += timedelta(days=n)
    if not parts:
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "confidence", "source"])
    out = pd.concat(parts, ignore_index=True)
    out["acq_date"] = pd.to_datetime(out["acq_date"]).dt.date
    return out


def _get_quota_aware(url: str, tries: int = 15, wait: float = 60):
    """FIRMS 每 10 分钟有请求额度，用完时返回 400。遇到就等一分钟再试，最多约 15 分钟。"""
    import time
    for _ in range(tries):
        r = http.get(url)
        if r.status_code == 200 and not r.text.lower().startswith(("invalid", "exceeding")):
            return r
        if r.status_code not in (400, 403, 429) and r.status_code != 200:
            r.raise_for_status()
        time.sleep(wait)
    raise RuntimeError(f"FIRMS 多次重试仍失败：{r.status_code} {r.text[:200]}")


def cluster_events(pts: pd.DataFrame, dist_km: float = 1.5, gap_days: int = 2,
                   min_points: int = 5, drop_low_conf: bool = True) -> pd.DataFrame:
    """时空聚类：相距 ≤ dist_km 且日期相差 ≤ gap_days 的火点算同一场火（连通分量）。"""
    p = pts.copy()
    if "type" in p:                      # 0=植被火；2=工厂等固定热源；3=海上
        p = p[pd.to_numeric(p["type"], errors="coerce").fillna(0) == 0]
    if drop_low_conf and "confidence" in p:
        p = p[p["confidence"].astype(str).str.lower().str[0] != "l"]
    if p.empty:
        return pd.DataFrame()
    p = p.reset_index(drop=True)
    lat0 = np.radians(p["latitude"].mean())
    x = np.radians(p["longitude"].values) * 6371 * np.cos(lat0)
    y = np.radians(p["latitude"].values) * 6371
    t = np.array([(d - date(2000, 1, 1)).days for d in p["acq_date"]])

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(np.c_[x, y]).query_pairs(dist_km, output_type="ndarray")
    if len(pairs):
        pairs = pairs[np.abs(t[pairs[:, 0]] - t[pairs[:, 1]]) <= gap_days]
    n = len(p)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) \
        else coo_matrix((n, n))
    _, lab = connected_components(g, directed=False)
    p["event"] = lab
    ev = p.groupby("event").agg(n_points=("latitude", "size"),
                                center_lat=("latitude", "mean"), center_lon=("longitude", "mean"),
                                ignition_date=("acq_date", "min"), end_date=("acq_date", "max"),
                                west=("longitude", "min"), east=("longitude", "max"),
                                south=("latitude", "min"), north=("latitude", "max"),
                                frp_sum=("frp", "sum") if "frp" in p else ("latitude", "size"))
    ev = ev[ev["n_points"] >= min_points].sort_values("ignition_date").reset_index(drop=True)
    ev["duration_days"] = [(b - a).days + 1 for a, b in zip(ev["ignition_date"], ev["end_date"])]
    return ev


def nearest_event(ev: pd.DataFrame, lat: float, lon: float) -> pd.Series | None:
    if ev.empty:
        return None
    d = np.hypot(ev["center_lat"] - lat, (ev["center_lon"] - lon) * np.cos(np.radians(lat)))
    return ev.loc[d.idxmin()]


def events_to_fires(ev: pd.DataFrame, province: str | None = None, tag: str = "VIIRS") -> pd.DataFrame:
    """把发现的事件变成火场名单行（名称自动生成，后续可人工改名）。"""
    rows = []
    for _, e in ev.iterrows():
        rows.append({"name_cn": f"{e['ignition_date'].year}{province or ''}{tag}_{e['ignition_date']:%m%d}_"
                                f"{e['center_lat']:.3f}N{e['center_lon']:.3f}E",
                     "province": province, "year": e["ignition_date"].year,
                     "ignition_date": e["ignition_date"], "end_date": e["end_date"],
                     "center_lat": round(float(e["center_lat"]), 5),
                     "center_lon": round(float(e["center_lon"]), 5),
                     "qc_notes": f"由 FIRMS 火点自动发现：{int(e['n_points'])} 个火点，持续 {int(e['duration_days'])} 天"})
    return pd.DataFrame(rows)
