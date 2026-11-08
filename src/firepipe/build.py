"""单个火场的完整处理：取数 → 质检 → 写栅格 → 采样 → 登记入库。"""
from __future__ import annotations

import json
import tempfile
import traceback
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  （注册 .rio 方法）
import pandas as pd

from . import burn, landcover, sentinel, terrain, weather
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
        qc["n_hotspots"] = int(len(pts))
        if pts.empty:
            return None
        qc["hotspot_dates"] = [str(min(pts["acq_date"])), str(max(pts["acq_date"]))]
        x, y = to_utm.transform(pts["longitude"].values, pts["latitude"].values)
        x, y = np.asarray(x), np.asarray(y)
        cols = inv.a * x + inv.b * y + inv.c
        rows = inv.d * x + inv.e * y + inv.f
        return np.c_[np.floor(rows).astype(int), np.floor(cols).astype(int)]
    except Exception as e:
        qc["issues"].append(f"火点读取失败：{e}")
        return None


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

    spec, smeta = sentinel.build_spectral(cfg, grid, fire)
    qc["sentinel"] = smeta
    for k in ("pre", "post"):
        if smeta[k]["valid_fraction"] < cfg["sentinel2"]["min_valid_fraction"]:
            qc["issues"].append(f"{k} 合成有效像元比例 {smeta[k]['valid_fraction']:.2f} 偏低")

    layers = {k: v.values.astype("float32") for k, v in spec.items()}
    template = spec["dnbr"]

    # 火烧范围与背景偏移
    from pyproj import Transformer
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{grid.epsg}", always_xy=True)
    inv = ~grid.geobox.transform
    hot_rc = _hotspots(cfg, grid, fire, to_utm, inv, qc)
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
    if qc["burned_area_ha"] < 10:
        qc["issues"].append("火烧面积过小（<10 公顷）")

    # 因子
    for k, v in terrain.build_terrain(cfg, grid).items():
        layers[k] = v.values.astype("float32")
    try:
        layers["worldcover"] = landcover.build_landcover(cfg, grid).values.astype("float32")
    except Exception as e:  # 土地覆盖缺失不阻断流程
        qc["issues"].append(f"WorldCover 读取失败：{e}")
    for k, v in weather.build_weather(cfg, grid, fire).items():
        layers[k] = v.values.astype("float32")

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


def _source(k: str) -> str:
    if k in {"elevation", "slope", "northness", "eastness", "tpi", "twi"}:
        return "cop-dem-glo-30"
    if k == "worldcover":
        return "ESA WorldCover 2021 v200"
    if k.startswith(("fire_", "prcp_", "dry_", "soilm")):
        return "ERA5-Land via Open-Meteo"
    return "sentinel-2-l2a (Earth Search)"


def process_queue(cfg: dict, db: DB, limit: int = 20) -> list[dict]:
    """处理所有“待处理且资料齐全、尚未用当前版本生成过数据”的火场。"""
    ver = cfg["pipeline_version"]
    fires = db.select("fires", {"qc_status": "eq.pending", "select": "*"})
    done = {a["fire_id"] for a in db.select("data_assets", {"variable": "eq.samples",
                                                            "pipeline_version": f"eq.{ver}",
                                                            "select": "fire_id"})}
    todo = [f for f in fires if f["fire_id"] not in done
            and all(f.get(k) for k in ("ignition_date", "end_date", "center_lat", "center_lon"))][:limit]
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
