"""Sentinel-2 L2A：逐像元去云、固定时间窗口中位数合成、计算 dNBR 和火前植被指数。

数据来源：Element 84 Earth Search（AWS 公开数据），不需要账号。
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import xarray as xr

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

BANDS = ["blue", "green", "red", "nir", "nir08", "swir16", "swir22"]


# ---------------- 光谱指数（输入为反射率 0–1） ----------------
def nbr(nir08, swir22):
    return (nir08 - swir22) / (nir08 + swir22)


def ndvi(nir, red):
    return (nir - red) / (nir + red)


def evi(nir, red, blue):
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)


def ndmi(nir08, swir16):
    return (nir08 - swir16) / (nir08 + swir16)


def severity_indices(nbr_pre, nbr_post):
    """dNBR、RdNBR、RBR（Key & Benson 2006；Miller & Thode 2007；Parks 2014）。"""
    dnbr = nbr_pre - nbr_post
    rdnbr = dnbr / np.sqrt(np.abs(nbr_pre)).clip(min=1e-3)
    rbr = dnbr / (nbr_pre + 1.001)
    return dnbr, rdnbr, rbr


# ---------------- 读取 ----------------
def _window(anchor: date, days: list[int]) -> str:
    a = anchor + timedelta(days=days[0])
    b = anchor + timedelta(days=days[1])
    return f"{a.isoformat()}/{b.isoformat()}"


def _scale_offset(item, band):
    rb = item.assets[band].extra_fields.get("raster:bands", [{}])[0]
    return rb.get("scale", 1e-4), rb.get("offset", 0.0)


def _fix_href(item):
    for a in item.assets.values():
        if a.href.startswith("s3://"):
            b, k = a.href[5:].split("/", 1)
            a.href = f"https://{b}.s3.amazonaws.com/{k}"
    return item


def search(cfg: dict, bbox, datetime_range: str, max_cloud: float | None = None):
    from pystac_client import Client
    c = Client.open(cfg["sentinel2"]["stac_url"])
    items = c.search(collections=[cfg["sentinel2"]["collection"]], bbox=bbox,
                     datetime=datetime_range,
                     query={"eo:cloud_cover": {"lt": max_cloud or cfg["sentinel2"]["max_scene_cloud"]}}).item_collection()
    return [_fix_href(i) for i in items]


def load_composite(cfg: dict, grid, datetime_range: str, fallback_range: str | None = None) -> tuple[xr.Dataset, dict]:
    """读取窗口内所有影像 → SCL 逐像元去云 → 反射率换算 → 中位数合成。
    标准窗口内没有影像时，用放宽的窗口和云量上限再找一次，并在质检记录里注明。"""
    import odc.stac

    items = search(cfg, grid.bbox_lonlat, datetime_range)
    relaxed = False
    if not items and fallback_range:
        items = search(cfg, grid.bbox_lonlat, fallback_range, max_cloud=95)
        relaxed = True
    meta = {"n_scenes": len(items), "dates": sorted({i.datetime.date().isoformat() for i in items}),
            "window_relaxed": relaxed}
    if not items:
        raise RuntimeError(f"{datetime_range} 内没有可用的 Sentinel-2 影像")

    # 不同处理基线的 scale/offset 可能不同 → 分组读取再换算
    groups: dict[tuple, list] = {}
    for it in items:
        key = tuple(_scale_offset(it, b) for b in BANDS)
        groups.setdefault(key, []).append(it)

    stacks = []
    valid_scl = cfg["sentinel2"]["scl_valid"]
    for key, its in groups.items():
        ds = odc.stac.load(its, bands=BANDS + ["scl"], geobox=grid.geobox, groupby="solar_day",
                           resampling={"scl": "nearest", "*": "average"})
        mask = ds["scl"].isin(valid_scl)
        refl = {}
        for b, (s, o) in zip(BANDS, key):
            v = ds[b].where(ds[b] > 0).astype("float32") * s + o
            refl[b] = v.where(mask & (v > 0) & (v < 1.5))
        stacks.append(xr.Dataset(refl))
    allds = xr.concat(stacks, dim="time") if len(stacks) > 1 else stacks[0]
    comp = allds.median("time", skipna=True)
    valid_frac = float(np.isfinite(comp["nir08"]).mean())
    meta["valid_fraction"] = valid_frac
    meta["n_obs_median"] = float(np.isfinite(allds["nir08"]).sum("time").median())
    return comp, meta


def build_spectral(cfg: dict, grid, fire: dict) -> tuple[dict[str, xr.DataArray], dict]:
    ign = fire["ignition_date"] if isinstance(fire["ignition_date"], date) else date.fromisoformat(str(fire["ignition_date"]))
    end = fire["end_date"] if isinstance(fire["end_date"], date) else date.fromisoformat(str(fire["end_date"]))
    pre, mpre = load_composite(cfg, grid, _window(ign, cfg["sentinel2"]["pre_window_days"]),
                               _window(ign, cfg["sentinel2"]["pre_window_fallback"]))
    post, mpost = load_composite(cfg, grid, _window(end, cfg["sentinel2"]["post_window_days"]),
                                 _window(end, cfg["sentinel2"]["post_window_fallback"]))

    nbr_pre = nbr(pre.nir08, pre.swir22)
    nbr_post = nbr(post.nir08, post.swir22)
    dnbr, rdnbr, rbr = severity_indices(nbr_pre, nbr_post)
    layers = {
        "nbr_pre": nbr_pre,
        "ndvi_pre": ndvi(pre.nir, pre.red),
        "evi_pre": evi(pre.nir, pre.red, pre.blue),
        "ndmi_pre": ndmi(pre.nir08, pre.swir16),
        "nbr_post": nbr_post,   # 仅用于生成标签，不作为预测因子
        "dnbr": dnbr, "rdnbr": rdnbr, "rbr": rbr,
    }
    return layers, {"pre": mpre, "post": mpost}
