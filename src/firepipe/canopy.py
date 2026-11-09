"""冠层高度：Meta / WRI 全球 1 m 冠层高度（Tolan et al. 2024），AWS 公开桶 dataforgood-fb-data，按 quadkey 分块的 COG。
通过 COG 金字塔直接按约 30 m 读取（不下载 1 m 全幅），再给 300 m 邻域的均值与标准差（林分结构是否均匀）。
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import ndimage

from . import http

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
_INDEX_CACHE: dict = {}


def _tiles_for_bbox(cfg: dict, bbox) -> list[str]:
    """从 tiles.geojson 找覆盖范围的 quadkey。索引约几 MB，进程内缓存。"""
    url = cfg["canopy"]["meta_index_url"]
    if url not in _INDEX_CACHE:
        r = http.get(url); r.raise_for_status()
        _INDEX_CACHE[url] = r.json()
    minx, miny, maxx, maxy = bbox
    out = []
    for f in _INDEX_CACHE[url]["features"]:
        coords = np.asarray(f["geometry"]["coordinates"][0], dtype=float)
        if coords.ndim == 3:
            coords = coords[0]
        tx0, ty0, tx1, ty1 = coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()
        if tx1 < minx or tx0 > maxx or ty1 < miny or ty0 > maxy:
            continue
        props = f.get("properties", {})
        tile = props.get("tile") or next((v for v in props.values() if isinstance(v, str) and v.isdigit()), None)
        if tile:
            out.append(tile)
    return out


def build_canopy(cfg: dict, grid) -> dict[str, np.ndarray]:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds

    res = cfg["grid"]["resolution_m"]
    gb = grid.geobox
    dst_crs = gb.crs.to_wkt()
    acc = np.zeros(gb.shape, "float64"); cnt = np.zeros(gb.shape, "float64")
    tiles = _tiles_for_bbox(cfg, grid.bbox_lonlat)
    if not tiles:
        raise RuntimeError("Meta 冠层高度索引里没有覆盖该范围的分块")
    for t in tiles:
        url = cfg["canopy"]["meta_tile_url"].format(tile=t)
        with rasterio.open(url) as src:
            # 用 WarpedVRT 直接重投影到网格，平均重采样会自动利用 COG 金字塔
            with WarpedVRT(src, crs=dst_crs, transform=gb.transform, width=gb.shape[1], height=gb.shape[0],
                           resampling=Resampling.average, nodata=np.nan, src_nodata=src.nodata) as vrt:
                a = vrt.read(1, masked=True).astype("float32").filled(np.nan)
        ok = np.isfinite(a) & (a >= 0) & (a < 150)
        acc[ok] += a[ok]; cnt[ok] += 1
    h = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype("float32")
    size = max(3, int(300 / res) | 1)
    v = np.nan_to_num(h, nan=0.0); w = np.isfinite(h).astype("float32")
    ws = ndimage.uniform_filter(w, size)
    mean = ndimage.uniform_filter(v, size) / np.maximum(ws, 1e-6)
    sq = ndimage.uniform_filter(v * v, size) / np.maximum(ws, 1e-6)
    sd = np.sqrt(np.maximum(sq - mean ** 2, 0))
    mean[ws < 0.2] = np.nan; sd[ws < 0.2] = np.nan
    return {"canopy_height": h, "canopy_h_mean300": mean.astype("float32"), "canopy_h_sd300": sd.astype("float32")}
