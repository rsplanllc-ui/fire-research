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


def load_composite(cfg: dict, grid, datetime_range: str, fallback_range: str | None = None,
                   bands: list[str] | None = None, max_cloud: float | None = None) -> tuple[xr.Dataset, dict]:
    """读取窗口内所有影像 → SCL 逐像元去云 → 反射率换算 → 中位数合成。
    标准窗口内没有影像时，用放宽的窗口和云量上限再找一次，并在质检记录里注明。"""
    import odc.stac

    bands = bands or BANDS
    items = search(cfg, grid.bbox_lonlat, datetime_range, max_cloud=max_cloud)
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
    ignore_off = cfg["sentinel2"].get("ignore_stac_offset", False)
    for it in items:
        key = tuple((s, 0.0 if ignore_off else o) for s, o in (_scale_offset(it, b) for b in bands))
        groups.setdefault(key, []).append(it)

    stacks = []
    valid_scl = cfg["sentinel2"]["scl_valid"]
    for key, its in groups.items():
        ds = odc.stac.load(its, bands=bands + ["scl"], geobox=grid.geobox, groupby="solar_day",
                           resampling={"scl": "nearest", "*": "average"})
        mask = ds["scl"].isin(valid_scl)
        refl = {}
        for b, (s, o) in zip(bands, key):
            v = ds[b].where(ds[b] > 0).astype("float32") * s + o     # DN=0 是无数据
            # 山地阴坡、冬季低太阳高度时，红/蓝波段大气校正后可能接近 0 或略小于 0，属正常暗像元：
            # 截到 0.0001 保留，而不是当作无效值丢掉（否则 NDVI/EVI 大面积缺失）
            refl[b] = v.where(mask & (v > -0.05) & (v < 1.5)).clip(min=1e-4)
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


# ---------------- 火前多年时序因子 ----------------
TS_BANDS = ["red", "nir", "nir08", "swir22"]


def build_timeseries(cfg: dict, grid, fire: dict, nbr_pre: xr.DataArray, ndvi_pre: xr.DataArray) -> tuple[dict, dict]:
    """火前 N 年的季节合成（夏季 6–8 月、冬季 12–2 月），得到：
    - evergreen_ratio：冬季 NDVI / 夏季 NDVI（针叶、常绿阔叶接近 1，落叶林低）
    - ndvi_amp：夏季与冬季 NDVI 之差（季节振幅）
    - nbr_3y_mean：多年 NBR 平均（长期生物量/湿度水平）
    - nbr_trend：NBR 随时间的斜率（每年），负值表示退化或干旱累积
    - nbr_pre_anom / ndvi_pre_anom：火前合成相对前几年同期的距平（当年旱情）
    任何一个窗口没影像就跳过该窗口，全部缺失时返回全 NaN 图层。"""
    ign = fire["ignition_date"] if isinstance(fire["ignition_date"], date) else date.fromisoformat(str(fire["ignition_date"]))
    ts = cfg.get("timeseries", {})
    years = int(ts.get("years", 3))
    cloud = ts.get("max_scene_cloud", 60)
    pre_days = cfg["sentinel2"]["pre_window_days"]
    comps = []            # (t_years_before, season, ds)
    meta = {"windows": []}
    for k in range(1, years + 1):
        y = ign.year - k
        for season, (a, b) in {"summer": (date(y, 6, 1), date(y, 8, 31)),
                               "winter": (date(y - 1, 12, 1), date(y, 2, 28))}.items():
            if b >= ign:
                continue
            rng = f"{a.isoformat()}/{b.isoformat()}"
            try:
                ds, m = load_composite(cfg, grid, rng, bands=TS_BANDS, max_cloud=cloud)
                t = -((ign - (a + (b - a) / 2)).days / 365.25)
                comps.append((t, season, ds)); meta["windows"].append({"range": rng, "n": m["n_scenes"]})
            except Exception as e:
                meta["windows"].append({"range": rng, "n": 0, "err": str(e)[:80]})
    # 前几年同期窗口（和火前窗口同一季节），用于距平
    same = []
    for k in range(1, years + 1):
        try:
            a = ign.replace(year=ign.year - k)
        except ValueError:
            a = ign.replace(year=ign.year - k, day=28)
        rng = _window(a, pre_days)
        try:
            ds, m = load_composite(cfg, grid, rng, bands=TS_BANDS, max_cloud=cloud)
            same.append(ds); meta["windows"].append({"range": rng, "n": m["n_scenes"], "same_season": True})
        except Exception as e:
            meta["windows"].append({"range": rng, "n": 0, "err": str(e)[:80], "same_season": True})

    shape = nbr_pre.shape
    nanarr = lambda: xr.full_like(nbr_pre, np.nan, dtype="float32")
    out = {k: nanarr() for k in ("evergreen_ratio", "ndvi_amp", "nbr_3y_mean", "nbr_trend", "nbr_pre_anom", "ndvi_pre_anom")}
    if comps:
        nd_s = [ndvi(d.nir, d.red) for t, s_, d in comps if s_ == "summer"]
        nd_w = [ndvi(d.nir, d.red) for t, s_, d in comps if s_ == "winter"]
        if nd_s and nd_w:
            S = xr.concat(nd_s, "t").mean("t", skipna=True); W = xr.concat(nd_w, "t").mean("t", skipna=True)
            out["evergreen_ratio"] = (W / S.clip(min=0.05)).clip(-1, 2).astype("float32")
            out["ndvi_amp"] = (S - W).astype("float32")
        nb = xr.concat([nbr(d.nir08, d.swir22) for t, s_, d in comps], "t")
        tt = np.array([t for t, s_, d in comps], "float32")
        out["nbr_3y_mean"] = nb.mean("t", skipna=True).astype("float32")
        if len(tt) >= 3:
            tm = tt - tt.mean()
            cov = ((nb - nb.mean("t")) * xr.DataArray(tm, dims="t")).sum("t", skipna=True)
            out["nbr_trend"] = (cov / float((tm ** 2).sum())).astype("float32")
    if same:
        ref_nbr = xr.concat([nbr(d.nir08, d.swir22) for d in same], "t").mean("t", skipna=True)
        ref_ndvi = xr.concat([ndvi(d.nir, d.red) for d in same], "t").mean("t", skipna=True)
        out["nbr_pre_anom"] = (nbr_pre - ref_nbr).astype("float32")
        out["ndvi_pre_anom"] = (ndvi_pre - ref_ndvi).astype("float32")
    meta["n_windows_ok"] = sum(1 for w in meta["windows"] if w["n"] > 0)
    return out, meta
