"""地形因子：Copernicus DEM GLO-30（AWS 公开数据）→ 高程、坡度、坡向、TPI、TWI。"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def load_dem(cfg: dict, grid):
    import odc.stac
    from pystac_client import Client

    from .sentinel import _fix_href
    c = Client.open(cfg["sentinel2"]["stac_url"])
    items = [_fix_href(i) for i in c.search(collections=[cfg["terrain"]["stac_collection"]],
                                            bbox=grid.bbox_lonlat).item_collection()]
    if not items:
        raise RuntimeError("没有找到 DEM")
    ds = odc.stac.load(items, bands=["data"], geobox=grid.geobox, resampling="bilinear")
    return ds["data"].max("time").astype("float32")


def slope_aspect(dem: np.ndarray, res: float):
    """返回坡度（度）和坡向（度，北=0，顺时针）。行向下=南。"""
    dzdy, dzdx = np.gradient(dem, res)          # dzdy: 沿行增加（向南）
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # 下坡方向 = -梯度；北分量 = +dzdy（行向南为正，所以向北下坡 = dzdy>0）
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360) % 360
    return slope, aspect


def tpi(dem: np.ndarray, res: float, radius_m: float):
    k = max(3, int(2 * radius_m / res) | 1)
    mean = ndimage.uniform_filter(np.nan_to_num(dem, nan=np.nanmean(dem)), size=k, mode="nearest")
    return dem - mean


def d8_accumulation(dem: np.ndarray) -> np.ndarray:
    """简单 D8 汇流累积（单位：像元数）。未做填洼，坑洼处汇流会中断，属已知近似。"""
    z = np.nan_to_num(dem, nan=np.nanmax(dem) + 1).astype("float64")
    n_r, n_c = z.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dist = np.array([np.hypot(dr, dc) for dr, dc in offs])
    pad = np.pad(z, 1, mode="edge")
    drops = np.stack([(z - pad[1 + dr:1 + dr + n_r, 1 + dc:1 + dc + n_c]) / d
                      for (dr, dc), d in zip(offs, dist)])
    best = drops.argmax(0)
    has_down = drops.max(0) > 0
    rr, cc = np.indices(z.shape)
    dr = np.array([o[0] for o in offs])[best]
    dc = np.array([o[1] for o in offs])[best]
    tr = np.clip(rr + dr, 0, n_r - 1)
    tc = np.clip(cc + dc, 0, n_c - 1)
    recv = np.where(has_down, tr * n_c + tc, -1).ravel()
    acc = np.ones(z.size)
    for i in np.argsort(-z.ravel(), kind="stable"):
        j = recv[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc.reshape(z.shape)


def twi(dem: np.ndarray, slope_deg: np.ndarray, res: float) -> np.ndarray:
    acc = d8_accumulation(dem)
    a = acc * res                                 # 单位等高线宽度的汇水面积
    tanb = np.tan(np.radians(slope_deg)).clip(min=1e-3)
    return np.log(a / tanb)


def build_terrain(cfg: dict, grid) -> dict:
    dem_da = load_dem(cfg, grid)
    res = float(cfg["grid"]["resolution_m"])
    dem = dem_da.values
    slope, aspect = slope_aspect(dem, res)
    layers = {
        "elevation": dem,
        "slope": slope,
        "northness": np.cos(np.radians(aspect)),
        "eastness": np.sin(np.radians(aspect)),
        "tpi": tpi(dem, res, cfg["terrain"]["tpi_radius_m"]),
        "twi": twi(dem, slope, res),
    }
    return {k: dem_da.copy(data=v.astype("float32")) for k, v in layers.items()}
