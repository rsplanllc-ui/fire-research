"""冠层高度：ETH Global Canopy Height 10 m（2020，Lang et al. 2023），3°×3° 分块 COG，与 WorldCover 同网格。
聚合到 30 m 取均值，另给 300 m 邻域的均值与标准差（林分结构是否均匀）。
"""
from __future__ import annotations

import os

import numpy as np
from scipy import ndimage

# ETH 的 libdrive 分享不响应 HEAD 请求，GDAL 需跳过；并允许按 Range 只读窗口
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff")

from .landcover import worldcover_tiles


def build_canopy(cfg: dict, grid) -> dict[str, np.ndarray]:
    import rioxarray  # noqa: F401
    import xarray as xr
    from rasterio.enums import Resampling
    from rioxarray.merge import merge_arrays

    arrs = []
    minx, miny, maxx, maxy = grid.bbox_lonlat
    for t in worldcover_tiles(grid.bbox_lonlat):
        url = cfg["canopy"]["eth_url"].format(tile=t)
        da = xr.open_dataarray("/vsicurl/" + url, engine="rasterio").squeeze(drop=True)
        arrs.append(da.rio.clip_box(minx, miny, maxx, maxy))
    da = merge_arrays(arrs) if len(arrs) > 1 else arrs[0]
    nod = da.rio.nodata
    da = da.astype("float32")
    if nod is not None:
        da = da.where(da != nod)
    da = da.where(da < 200)                                   # 255 = 无数据
    out = da.rio.reproject(grid.geobox.crs.to_wkt(), shape=grid.geobox.shape,
                           transform=grid.geobox.transform, resampling=Resampling.average, nodata=np.nan)
    h = out.values.astype("float32")
    res = cfg["grid"]["resolution_m"]
    size = max(3, int(300 / res) | 1)
    v = np.nan_to_num(h, nan=0.0); w = np.isfinite(h).astype("float32")
    ws = ndimage.uniform_filter(w, size)
    mean = ndimage.uniform_filter(v, size) / np.maximum(ws, 1e-6)
    sq = ndimage.uniform_filter(v * v, size) / np.maximum(ws, 1e-6)
    sd = np.sqrt(np.maximum(sq - mean ** 2, 0))
    mean[ws < 0.2] = np.nan; sd[ws < 0.2] = np.nan
    return {"canopy_height": h, "canopy_h_mean300": mean.astype("float32"), "canopy_h_sd300": sd.astype("float32")}
