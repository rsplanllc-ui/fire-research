"""单个火场的完整处理：取数 → 质检 → 写栅格 → 采样 → 登记入库。"""
from __future__ import annotations

import json
import tempfile
import zlib
import traceback
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  （注册 .rio 方法）
import pandas as pd

from scipy import ndimage

from . import burn, canopy, fireobs, landcover, sentinel, terrain, weather
from .db import DB, sha256
from .geo import FireGrid

LABEL_ONLY = {"nbr_post", "dnbr", "rdnbr", "rbr", "dnbr_offset", "perimeter"}


def _boundary(db: DB | None, fire: dict, workdir: Path):
    if not fire.get("boundary_path") or db is None:
        return None
    import shapely.geometry as sg
    p = db.download(fire["boundary_path"], workdir / "boundary.geojson")
    gj = json.loads(p.read_text(encoding="utf-8"))
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    geom = sg.shape(feats[0]["geometry"]) if len(feats) == 1 else \
        sg.MultiPolygon([sg.shape(f["geometry"]) for f in feats]).buffer(0)
    return geom


def _rasterize(geom, grid) -> np.ndarray:
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely.ops import transform
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{grid.epsg}", always_xy=True).transform
    g = transform(tr, geom)
    return rasterize([(g, 1)], out_shape=grid.geobox.shape, transform=grid.geobox.transform,
                     fill=0, dtype="uint8").astype(bool)


def _hotspots(cfg, grid, fire, to_utm, inv, qc):
    """取火灾期间（前后各放宽几天）的 VIIRS 火点，换算成网格行列号。没有密钥或失败时返回 None。"""
    import os
    from datetime import date, timedelta

    from . import firms
    if not os.environ.get("FIRMS_MAP_KEY"):
        return None
    try:
        a = date.fromisoformat(str(fire["ignition_date"])) - timedelta(days=cfg["burn"]["hotspot_pad_days"])
        b = date.fromisoformat(str(fire["end_date"])) + timedelta(days=cfg["burn"]["hotspot_pad_days"])
        pts = firms.fetch(grid.bbox_lonlat, a, b)
        if "confidence" in pts:
            pts = pts[pts["confidence"].astype(str).str.lower().str[0] != "l"]
        if "type" in pts:
            pts = pts[pd.to_numeric(pts["type"], errors="coerce").fillna(0) == 0]
        qc["n_hotspots"] = int(len(pts))
        if pts.empty:
            return None
        qc["hotspot_dates"] = [str(min(pts["acq_date"])), str(max(pts["acq_date"]))]
        x, y = to_utm.transform(pts["longitude"].values, pts["latitude"].values)
        x, y = np.asarray(x), np.asarray(y)
        cols = inv.a * x + inv.b * y + inv.c
        rows = inv.d * x + inv.e * y + inv.f
        out = pd.DataFrame({"row": np.floor(rows).astype(int), "col": np.floor(cols).astype(int),
                            "acq_date": pts["acq_date"].values,
                            "acq_time": pd.to_numeric(pts.get("acq_time", pd.Series(0, index=pts.index)),
                                                      errors="coerce").fillna(0).values,
                            "frp": pd.to_numeric(pts.get("frp", pd.Series(np.nan, index=pts.index)),
                                                 errors="coerce").values})
        return out
    except Exception as e:      # 火点取不到：让这场处理失败，下次自动重试，而不是退回不可靠的方法
        raise RuntimeError(f"火点读取失败：{e}") from e


def sample_pixels(layers: dict[str, np.ndarray], perimeter: np.ndarray, target: str,
                  spacing: int, max_n: int, seed: int) -> pd.DataFrame:
    """在火场范围内按固定间距取点，再随机抽到上限。"""
    keep = np.zeros_like(perimeter)
    keep[::spacing, ::spacing] = True
    m = perimeter & keep & np.isfinite(layers[target])
    rr, cc = np.nonzero(m)
    if len(rr) > max_n:
        rng = np.random.default_rng(seed)
        i = rng.choice(len(rr), max_n, replace=False)
        rr, cc = rr[i], cc[i]
    df = pd.DataFrame({"row": rr, "col": cc})
    for k, v in layers.items():
        if k != "perimeter":
            df[k] = v[rr, cc]
    return df


def process_fire(cfg: dict, fire: dict, db: DB | None = None, out_dir: str | None = None) -> dict:
    """返回质检摘要。db=None 时只在本地生成文件（用于测试）。"""
    ver = cfg["pipeline_version"]
    workdir = Path(out_dir or tempfile.mkdtemp(prefix=f"{fire['fire_id']}_"))
    workdir.mkdir(parents=True, exist_ok=True)
    qc: dict = {"fire_id": fire["fire_id"], "pipeline_version": ver, "issues": [], "notes": []}

    geom = _boundary(db, fire, workdir)
    grid = FireGrid.from_fire(fire, cfg["grid"]["resolution_m"], cfg["grid"]["bbox_half_deg"],
                              boundary_bounds=geom.bounds if geom is not None else None,
                              buffer_m=cfg["grid"]["boundary_buffer_m"])
    res = float(cfg["grid"]["resolution_m"])

    from pyproj import Transformer
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{grid.epsg}", always_xy=True)
    inv = ~grid.geobox.transform
    hs = _hotspots(cfg, grid, fire, to_utm, inv, qc)
    hot_rc = hs[["row", "col"]].to_numpy() if hs is not None else None
    wc = None
    try:
        wc = landcover.build_landcover(cfg, grid).values.astype("float32")
    except Exception as e:  # 土地覆盖缺失不阻断流程
        qc["issues"].append(f"WorldCover 读取失败：{e}")
    if wc is not None and hot_rc is not None and len(hot_rc):
        r = np.clip(hot_rc[:, 0], 0, wc.shape[0] - 1)
        c = np.clip(hot_rc[:, 1], 0, wc.shape[1] - 1)
        near = ndimage.binary_dilation(np.zeros(wc.shape, bool) | _mark(wc.shape, r, c), iterations=8)
        qc["forest_share_hotspots"] = float(np.mean(wc[near] == 10))
        if qc["forest_share_hotspots"] < cfg["qc"]["min_forest_share"] * 0.6:
            qc["issues"].append(f"非森林火：火点周围林地占比 {qc['forest_share_hotspots']:.2f}")
            return _finish_early(db, fire, qc)

    spec, smeta = sentinel.build_spectral(cfg, grid, fire)
    qc["sentinel"] = smeta
    for k in ("pre", "post"):
        if smeta[k].get("window_relaxed"):
            qc["notes"].append(f"{k} 影像窗口已放宽")
        if smeta[k]["valid_fraction"] < cfg["sentinel2"]["min_valid_fraction"]:
            qc["issues"].append(f"{k} 合成有效像元比例 {smeta[k]['valid_fraction']:.2f} 偏低")

    layers = {k: v.values.astype("float32") for k, v in spec.items()}
    template = spec["dnbr"]
    # 火前多年时序因子（v1.7）
    try:
        tsl, tsm = sentinel.build_timeseries(cfg, grid, fire, spec["nbr_pre"], spec["ndvi_pre"])
        layers.update({k: v.values.astype("float32") for k, v in tsl.items()})
        qc["timeseries"] = {"n_windows_ok": tsm["n_windows_ok"], "n_windows": len(tsm["windows"])}
        if tsm["n_windows_ok"] < 3:
            qc["notes"].append(f"多年时序窗口只有 {tsm['n_windows_ok']} 个有影像")
    except Exception as e:
        qc["notes"].append(f"多年时序因子失败：{e}")

    # 火烧范围与背景偏移
    if geom is not None:
        perim = _rasterize(geom, grid)
        qc["perimeter_source"] = "boundary"
        ring = burn.offset_ring(perim, res, cfg["burn"]["unburned_ring_m"])
        off = burn.dnbr_offset(layers["dnbr"], ring)
    elif hot_rc is not None and len(hot_rc) >= cfg["burn"]["min_hotspots"]:
        perim, off, _ = burn.hotspot_perimeter(layers["dnbr"], hot_rc, res, cfg["burn"]["dnbr_threshold"],
                                               cfg["burn"]["hotspot_near_m"], cfg["burn"]["background_far_m"],
                                               cfg["burn"]["min_patch_pixels"])
        qc["perimeter_source"] = "viirs_hotspots+dnbr"
        qc["notes"].append("火烧范围由火点+dNBR 自动提取，建议抽查")
    else:
        x, y = to_utm.transform(float(fire["center_lon"]), float(fire["center_lat"]))
        c, r = inv @ (x, y)
        perim = burn.derive_perimeter(layers["dnbr"], (int(r), int(c)),
                                      cfg["burn"]["dnbr_threshold"], cfg["burn"]["min_patch_pixels"])
        qc["perimeter_source"] = "derived_dnbr"
        qc["issues"].append("没有火点，火烧范围仅按 dNBR 阈值提取，可靠性低")
        ring = burn.offset_ring(perim, res, cfg["burn"]["unburned_ring_m"])
        off = burn.dnbr_offset(layers["dnbr"], ring)
    layers["dnbr_offset"] = layers["dnbr"] - off
    layers["perimeter"] = perim.astype("float32")
    qc["dnbr_offset_value"] = off
    qc["burned_area_ha"] = float(perim.sum() * res * res / 1e4)

    # v1.6 质检：零散火、云遮挡、低烈度、无火点
    if qc["perimeter_source"] == "derived_dnbr" and cfg["qc"].get("require_hotspots", True):
        qc["issues"].append("没有火点，排除")
        return _finish_early(db, fire, qc)
    if perim.any():
        lab, n = ndimage.label(perim)
        sizes = np.bincount(lab.ravel())[1:]
        qc["n_patches"] = int(n)
        qc["largest_patch_share"] = float(sizes.max() / sizes.sum())
        qc["largest_patch_ha"] = float(sizes.max() * res * res / 1e4)
        if (qc["largest_patch_share"] < cfg["qc"].get("min_largest_patch_share", 0)
                and qc["largest_patch_ha"] < cfg["qc"].get("min_largest_patch_ha", 0)):
            qc["issues"].append(f"零散多点火：最大斑块只占 {qc['largest_patch_share']:.2f}、"
                                f"{qc['largest_patch_ha']:.0f} 公顷（{n} 个斑块）")
        inside = layers["dnbr_offset"][perim]
        qc["dnbr_mean_perim"] = float(np.nanmean(inside))
        if qc["dnbr_mean_perim"] < cfg["qc"].get("min_dnbr_mean", 0):
            qc["issues"].append(f"低烈度火：范围内平均 dNBR {qc['dnbr_mean_perim']:.2f}")
    if hot_rc is not None and len(hot_rc):
        r = np.clip(hot_rc[:, 0], 0, perim.shape[0] - 1); c = np.clip(hot_rc[:, 1], 0, perim.shape[1] - 1)
        near = ndimage.binary_dilation(_mark(perim.shape, r, c),
                                       iterations=max(1, int(cfg["burn"]["hotspot_near_m"] / res)))
        qc["valid_near_hotspots"] = float(np.mean(np.isfinite(layers["dnbr"][near])))
        if qc["valid_near_hotspots"] < cfg["qc"].get("min_valid_near_hotspots", 0):
            qc["issues"].append(f"火点周边云遮挡：有效像元比例 {qc['valid_near_hotspots']:.2f}")
    if qc["burned_area_ha"] < cfg["qc"]["min_area_ha"]:
        qc["issues"].append(f"火烧面积过小（<{cfg['qc']['min_area_ha']} 公顷）")

    # 因子
    for k, v in terrain.build_terrain(cfg, grid).items():
        layers[k] = v.values.astype("float32")
    if wc is not None:
        layers["worldcover"] = wc
        if perim.any():
            qc["forest_share"] = float(np.mean(wc[perim] == 10))
            if qc["forest_share"] < cfg["qc"]["min_forest_share"]:
                qc["issues"].append(f"非森林火：火烧范围内林地占比 {qc['forest_share']:.2f}")
    try:
        layers.update(canopy.build_canopy(cfg, grid))
    except Exception as e:
        qc["notes"].append(f"冠层高度读取失败：{e}")
    for k, v in weather.build_weather(cfg, grid, fire).items():
        layers[k] = v.values.astype("float32")
    layers.update(fireobs.context_layers(layers, res))
    # 火前可得的风坡因子：用火灾期间主导风向（预测时相当于天气预报）
    aspect = (np.degrees(np.arctan2(layers["eastness"], layers["northness"])) + 360) % 360
    if "fire_winddir" in layers:
        ang = fireobs._ang_diff((aspect + 180) % 360, (layers["fire_winddir"] + 180) % 360)
        layers["aspect_firewind_angle"] = ang
        layers["firewind_slope_index"] = (layers["fire_windmax"] * np.cos(np.radians(ang))
                                          * np.tan(np.radians(layers["slope"]))).astype("float32")
        # 迎风暴露：山脊（TPI>0）且面向来风为正，背风谷地为负
        tpi_n = layers["tpi"] / (np.nanstd(layers["tpi"]) + 1e-6)
        layers["windward_exposure"] = (tpi_n * np.cos(np.radians(ang))).astype("float32")

    # 火中观测（火点）+ 燃烧当天天气
    from datetime import date as _date
    ign = _date.fromisoformat(str(fire["ignition_date"]))
    if hs is not None:
        layers.update(fireobs.hotspot_layers(hs, perim.shape, res, ign, cfg["burn"]["hotspot_near_m"]))
        try:
            layers.update(fireobs.burn_day_weather(layers["burn_day"], weather.fire_daily(cfg, fire), ign))
        except Exception as e:
            qc["notes"].append(f"燃烧当天天气读取失败：{e}")
        layers.update(fireobs.fire_behavior_layers(
            hs, perim.shape, res, ign, aspect, layers["slope"],
            layers.get("bd_winddir"), layers.get("bd_windmax"), cfg["burn"].get("spread_search_m", 3000)))

    shapes = {k: v.shape for k, v in layers.items()}
    if len(set(shapes.values())) != 1:
        raise RuntimeError(f"图层尺寸不一致：{shapes}")

    # 写栅格（COG）+ 登记
    assets = []
    for k, arr in layers.items():
        path = workdir / f"{k}.tif"
        template.copy(data=arr).rio.write_crs(f"EPSG:{grid.epsg}").rio.to_raster(
            path, driver="COG", compress="DEFLATE")
        remote = f"processed/{ver}/{fire['fire_id']}/{k}.tif"
        if db:
            db.upload(path, remote, "image/tiff")
        assets.append({"fire_id": fire["fire_id"], "variable": k, "storage_path": remote,
                       "resolution_m": res, "pipeline_version": ver,
                       "source": _source(k), "checksum_sha256": sha256(path),
                       "qc_status": "ok" if not qc["issues"] else "pending"})

    # 采样表
    s = cfg["sample"]
    df = sample_pixels(layers, perim, cfg["model"]["target"], s["min_spacing_px"],
                       s["max_per_fire"], s["seed"])
    df.insert(0, "fire_id", fire["fire_id"])
    spath = workdir / "samples.parquet"
    df.to_parquet(spath, index=False)
    sremote = f"samples/{ver}/{fire['fire_id']}.parquet"
    qc["n_samples"] = len(df)
    if len(df) < cfg["qc"]["min_samples"]:
        qc["issues"].append(f"样本过少（{len(df)}）")
    qc["dnbr_mean"] = float(df[cfg["model"]["target"]].mean()) if len(df) else None
    qc["dnbr_std"] = float(df[cfg["model"]["target"]].std()) if len(df) else None
    (workdir / "qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2, default=str))

    if db:
        db.upload(spath, sremote)
        db.upload(workdir / "qc.json", f"processed/{ver}/{fire['fire_id']}/qc.json", "application/json")
        assets.append({"fire_id": fire["fire_id"], "variable": "samples", "storage_path": sremote,
                       "pipeline_version": ver, "source": "derived", "checksum_sha256": sha256(spath),
                       "qc_status": "ok" if not qc["issues"] else "pending"})
        db.upsert("data_assets", assets, on_conflict="fire_id,variable,pipeline_version")
        db.update("fires", {"fire_id": fire["fire_id"]},
                  {"qc_status": "pending" if qc["issues"] else "ok",
                   "qc_notes": "；".join(qc["issues"] + qc["notes"]) or None})
    return qc


def _mark(shape, r, c):
    m = np.zeros(shape, bool)
    m[r, c] = True
    return m


def _finish_early(db, fire, qc):
    if db:
        db.update("fires", {"fire_id": fire["fire_id"]},
                  {"qc_status": "excluded", "qc_notes": "；".join(qc["issues"])})
    return qc


def _source(k: str) -> str:
    if k.startswith(("hs_", "burn_day")):
        return "NASA FIRMS VIIRS 375 m"
    if k.startswith("bd_"):
        return "ERA5-Land via Open-Meteo（燃烧当天）"
    if k.startswith("spread_") or k in {"aspect_wind_angle", "wind_slope_index"}:
        return "火点时序 + DEM + 燃烧当天风向（派生）"
    if k in {"aspect_firewind_angle", "firewind_slope_index", "windward_exposure"}:
        return "DEM + 火灾期间主导风向（派生）"
    if k in {"evergreen_ratio", "ndvi_amp", "nbr_3y_mean", "nbr_trend", "nbr_pre_anom", "ndvi_pre_anom"}:
        return "sentinel-2-l2a 火前多年季节合成（派生）"
    if k in {"elevation", "slope", "northness", "eastness", "tpi", "twi"}:
        return "cop-dem-glo-30"
    if k == "worldcover":
        return "ESA WorldCover 2021 v200"
    if k.startswith("canopy_"):
        return "Meta/WRI Global Canopy Height 1 m (Tolan et al. 2024)"
    if k.startswith(("fire_", "prcp_", "dry_", "soilm")):
        return "ERA5-Land via Open-Meteo"
    return "sentinel-2-l2a (Earth Search)"


def process_queue(cfg: dict, db: DB, limit: int = 20, shard: int = 0, nshards: int = 1,
                  split: str | None = None) -> list[dict]:
    """处理所有“待处理且资料齐全、尚未用当前版本生成过数据”的火场。"""
    ver = cfg["pipeline_version"]
    from . import catalog
    allf = pd.DataFrame(db.select("fires", {"qc_status": "neq.excluded", "select": "*"}))
    if not allf.empty:
        for keep, drop, km in catalog.find_duplicates(allf, cfg['discovery'].get('dedupe_km', 3)):
            db.update("fires", {"fire_id": drop},
                      {"qc_status": "excluded",
                       "qc_notes": f"与 {keep} 为同一场火（中心相距 {km:.2f} km、日期重叠），已排除避免重复"})
    fires = db.select("fires", {"qc_status": "in.(pending,ok)", "select": "*"})   # 版本升级后“已通过”的也要重做
    done = {a["fire_id"] for a in db.select("data_assets", {"variable": "eq.samples",
                                                            "pipeline_version": f"eq.{ver}",
                                                            "select": "fire_id"})}
    if split == "none":                         # 只处理还没划分的新火场
        fires = [f for f in fires if not f.get("split")]
    elif split:                                 # 只处理某个划分（如 dev），省算力；测试集留到要用时再做
        fires = [f for f in fires if f.get("split") == split]
    todo = [f for f in fires if f["fire_id"] not in done
            and all(f.get(k) for k in ("ignition_date", "end_date", "center_lat", "center_lon"))
            and zlib.crc32(f["fire_id"].encode()) % nshards == shard][:limit]
    results = []
    for f in todo:
        rid = db.run_start("process", f["fire_id"])
        try:
            qc = process_fire(cfg, f, db)
            db.run_end(rid, "done", json.dumps(qc, ensure_ascii=False, default=str))
            results.append(qc)
        except Exception:
            db.run_end(rid, "failed", traceback.format_exc())
            db.update("fires", {"fire_id": f["fire_id"]}, {"qc_notes": "处理失败，见 pipeline_runs"})
    return results
