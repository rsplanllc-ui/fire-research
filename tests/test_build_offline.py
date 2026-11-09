"""不联网测试单火场流程：用合成图层替代远程数据，检查范围提取、偏移校正、写 COG、采样。"""
import numpy as np
import pandas as pd

from firepipe import build, landcover, sentinel, terrain, weather
from firepipe.config import load_config
from firepipe.geo import FireGrid


def test_process_fire_offline(monkeypatch, tmp_path):
    import copy
    cfg = copy.deepcopy(load_config())
    cfg["grid"]["bbox_half_deg"] = 0.03
    cfg["qc"]["require_hotspots"] = False      # 离线测试没有火点，仍走阈值法
    fire = {"fire_id": "HN_2022_999", "center_lat": 26.0, "center_lon": 113.0,
            "ignition_date": "2022-10-17", "end_date": "2022-10-20"}
    grid = FireGrid.from_fire(fire, 30, 0.03)
    shape = grid.geobox.shape
    coords = grid.geobox.coordinates
    import xarray as xr

    def da(v):
        return xr.DataArray(v.astype("float32"), coords={"y": coords["y"].values, "x": coords["x"].values},
                            dims=("y", "x"))

    rng = np.random.default_rng(0)
    r, c = np.indices(shape)
    cy, cx = shape[0] // 2, shape[1] // 2
    fire_mask = np.hypot(r - cy, c - cx) < 40
    nbr_pre = 0.6 + rng.normal(0, 0.02, shape)
    sev = np.where(fire_mask, 0.3 + 0.3 * np.sin(r / 7), 0)
    nbr_post = nbr_pre - sev - 0.03                       # 0.03 = 季节性偏移

    def fake_spectral(cfg, grid, fire):
        dnbr, rd, rb = sentinel.severity_indices(nbr_pre, nbr_post)
        lay = {"nbr_pre": da(nbr_pre), "ndvi_pre": da(nbr_pre), "evi_pre": da(nbr_pre * 0.6),
               "ndmi_pre": da(nbr_pre * 0.5), "nbr_post": da(nbr_post),
               "dnbr": da(dnbr), "rdnbr": da(rd), "rbr": da(rb)}
        meta = {"valid_fraction": 0.95, "n_scenes": 3}
        return lay, {"pre": meta, "post": meta}

    def fake_terrain(cfg, grid):
        dem = 300 + r * 2.0
        sl, asp = terrain.slope_aspect(dem, 30)
        return {"elevation": da(dem), "slope": da(sl), "northness": da(np.cos(np.radians(asp))),
                "eastness": da(np.sin(np.radians(asp))), "tpi": da(terrain.tpi(dem, 30, 300)),
                "twi": da(terrain.twi(dem, sl, 30))}

    monkeypatch.setattr(sentinel, "build_spectral", fake_spectral)
    monkeypatch.setattr(sentinel, "build_timeseries", lambda cfg, grid, fire, a, b: ({"evergreen_ratio": da(np.full(shape, 0.9))}, {"n_windows_ok": 6, "windows": [1] * 6}))
    monkeypatch.setattr(terrain, "build_terrain", fake_terrain)
    monkeypatch.setattr(landcover, "build_landcover", lambda cfg, grid: da(np.full(shape, 10.0)))
    pts = pd.DataFrame({"lon": [112.97, 113.03], "lat": [25.97, 26.03],
                        "fire_tmax": [30.0, 32.0], "prcp_30d": [10.0, 12.0]})
    monkeypatch.setattr(weather, "build_weather", lambda cfg, grid, fire: weather.idw_to_grid(pts, grid))

    qc = build.process_fire(cfg, fire, db=None, out_dir=str(tmp_path))
    assert qc["perimeter_source"] == "derived_dnbr"
    assert abs(qc["dnbr_offset_value"] - 0.03) < 0.01       # 偏移被正确估出
    assert 0.3 < qc["burned_area_ha"] / (np.pi * 40 ** 2 * 0.09) < 1.3
    s = pd.read_parquet(tmp_path / "samples.parquet")
    assert 100 < len(s) <= cfg["sample"]["max_per_fire"]
    assert {"dnbr_offset", "slope", "fire_tmax", "worldcover"} <= set(s.columns)
    assert (tmp_path / "dnbr_offset.tif").exists()
