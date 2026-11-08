"""ESA WorldCover 2021（10 m，AWS 公开数据）→ 按众数聚合到 30 m 网格。"""
from __future__ import annotations

import math


def worldcover_tiles(bbox) -> list[str]:
    minx, miny, maxx, maxy = bbox
    tiles = []
    for lat in range(int(math.floor(miny / 3) * 3), int(math.floor(maxy / 3) * 3) + 1, 3):
        for lon in range(int(math.floor(minx / 3) * 3), int(math.floor(maxx / 3) * 3) + 1, 3):
            ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
            tiles.append(ns + ew)
    return tiles


def build_landcover(cfg: dict, grid):
    import rioxarray  # noqa: F401
    import xarray as xr
    from rasterio.enums import Resampling

    arrs = []
    for t in worldcover_tiles(grid.bbox_lonlat):
        url = cfg["landcover"]["worldcover_url"].format(tile=t)
        da = xr.open_dataarray(url, engine="rasterio").squeeze(drop=True)
        minx, miny, maxx, maxy = grid.bbox_lonlat
        da = da.rio.clip_box(minx, miny, maxx, maxy)
        arrs.append(da)
    from rioxarray.merge import merge_arrays
    da = merge_arrays(arrs) if len(arrs) > 1 else arrs[0]
    out = da.rio.reproject(grid.geobox.crs.to_wkt(), shape=grid.geobox.shape,
                           transform=grid.geobox.transform, resampling=Resampling.mode)
    return out.astype("float32").where(out > 0)


# 用于把类别变量做成哑变量时参考
WORLDCOVER_CLASSES = {10: "林地", 20: "灌木", 30: "草地", 40: "农田", 50: "建筑",
                      60: "裸地", 80: "水体", 90: "湿地", 95: "红树林"}
