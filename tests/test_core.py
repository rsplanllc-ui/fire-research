import numpy as np
import pandas as pd
import pytest

from firepipe import burn, catalog, terrain
from firepipe.config import load_config
from firepipe.metrics import per_fire, r2, summarize
from firepipe.model import grouped_cv, run_dev
from firepipe.sentinel import nbr, severity_indices


def test_r2_can_be_negative_and_not_corr_squared():
    y = np.array([0.1, 0.2, 0.3, 0.4])
    assert r2(y, y) == pytest.approx(1.0)
    assert r2(y, y + 1.0) < 0           # 完全相关但整体偏移 → R² 为负
    assert np.isnan(r2([1, 1, 1], [1, 2, 3]))


def test_per_fire_metrics():
    df = pd.DataFrame({"fire_id": ["a"] * 3 + ["b"] * 3,
                       "y": [0, 1, 2, 0, 1, 2], "pred": [0, 1, 2, 1, 1, 1]})
    fm = per_fire(df).set_index("fire_id")
    assert fm.loc["a", "r2"] == pytest.approx(1)
    assert fm.loc["b", "r2"] == pytest.approx(0)
    s = summarize(fm.reset_index(), threshold=0.9)
    assert s["share_fires_ge_threshold"] == 0.5 and not s["all_fires_ge_threshold"]


def test_indices():
    pre, post = nbr(0.30, 0.10), nbr(0.15, 0.25)
    d, rd, rb = severity_indices(np.array(pre), np.array(post))
    assert d == pytest.approx(0.5 - (-0.25))
    assert rb == pytest.approx(d / (pre + 1.001))


def test_parse_and_ids():
    df = pd.DataFrame({"name_cn": ["2022湖南衡阳祁东县3", "2022湖南衡阳祁东县2", "2018广东省揭阳市普宁市"]})
    out = catalog.prepare(df)
    assert set(out["fire_id"]) == {"HN_2022_001", "HN_2022_002", "GD_2018_001"}
    assert (out["qc_status"] == "excluded").all()        # 没有日期坐标 → 不处理


def test_split_is_stratified_and_reproducible():
    fires = pd.DataFrame({"fire_id": [f"F{i:03d}" for i in range(50)],
                          "province": ["湖南"] * 30 + ["福建"] * 20})
    s1 = catalog.stratified_split(fires, 0.2, ["province"], 7)
    s2 = catalog.stratified_split(fires, 0.2, ["province"], 7)
    assert (s1 == s2).all()
    assert (s1[:30] == "test").sum() == 6 and (s1[30:] == "test").sum() == 4


def test_slope_aspect_directions():
    res = 30.0
    rows = np.arange(10)[:, None] * np.ones((1, 10))
    dem_south_low = 100 - rows * 3      # 向南（行号增大）变低 → 朝南坡
    _, asp = terrain.slope_aspect(dem_south_low, res)
    assert np.allclose(asp[2:-2, 2:-2], 180)
    cols = np.ones((10, 1)) * np.arange(10)[None, :]
    _, asp_e = terrain.slope_aspect(100 - cols * 3, res)   # 向东变低 → 朝东坡
    assert np.allclose(asp_e[2:-2, 2:-2], 90)


def test_flow_accumulation_valley():
    x = np.abs(np.arange(-5, 6))[None, :] * 1.0
    dem = x + np.arange(11)[:, None] * 0.1     # V 形山谷，向下游（行号增大）缓降
    dem = dem[::-1]
    acc = terrain.d8_accumulation(dem)
    assert acc[:, 5].max() == acc.max()          # 谷底汇流最大
    assert acc.max() > 20


def test_perimeter_keeps_center_patch_and_fills_holes():
    d = np.zeros((60, 60))
    d[10:40, 10:40] = 0.5
    d[20:23, 20:23] = 0.0          # 火场内的低烧毁岛
    d[50:55, 50:55] = 0.5          # 远处的另一块变化
    p = burn.derive_perimeter(d, (25, 25), 0.1, 10)
    assert p[21, 21] and p[15, 15] and not p[52, 52]
    ring = burn.offset_ring(p, 30, [60, 300])
    assert not (ring & p).any()


def test_grouped_cv_never_leaks_fire():
    rng = np.random.default_rng(0)
    rows = []
    for f in range(8):
        x = rng.normal(size=200)
        rows.append(pd.DataFrame({"fire_id": f"F{f}", "slope": x, "dnbr_offset": 0.3 * x + f * 0.1}))
    df = pd.concat(rows)
    cfg = load_config()
    cfg = {**cfg, "model": {**cfg["model"], "features": {"terrain": ["slope"]}}}
    out = run_dev(cfg, df, ["terrain"], ["mean_baseline", "ridge"])
    assert out["ridge"]["summary"]["r2_median"] > out["mean_baseline"]["summary"]["r2_median"]
    oof = grouped_cv(df, ["slope"], "dnbr_offset", "ridge", 4)
    assert oof["pred"].notna().all()


def test_upsert_normalizes_keys(monkeypatch):
    import json as _json

    from firepipe import db as dbmod
    sent = {}

    class R:
        ok = True
        def json(self):
            return []

    def fake_post(url, headers=None, params=None, data=None, timeout=None):
        sent["rows"] = _json.loads(data)
        return R()

    monkeypatch.setattr(dbmod.requests, "post", fake_post)
    d = dbmod.DB("https://x.supabase.co\n", " k\n")
    assert d.url == "https://x.supabase.co"
    d.upsert("data_assets", [{"a": 1, "b": 2}, {"a": 3}])
    assert all(set(r) == {"a", "b"} for r in sent["rows"])


def test_hotspot_perimeter_removes_drought_background():
    rng = np.random.default_rng(3)
    d = 0.25 + rng.normal(0, 0.02, (200, 200))          # 大旱：全区 dNBR 普遍抬高 0.25
    d[80:120, 80:130] += 0.4                             # 真实火烧区
    d[10:30, 150:190] += 0.4                             # 远处另一块变化（没有火点，如采伐）
    hot = np.array([[90, 90], [100, 110], [115, 125]])
    p, bg, _ = burn.hotspot_perimeter(d, hot, 30, 0.10, near_m=900, far_m=2000, min_patch=20)
    assert abs(bg - 0.25) < 0.02
    assert p[100, 100] and not p[20, 170] and not p[160, 40]
    assert 0.7 < p.sum() / (40 * 50) < 1.3


def test_http_retries_then_succeeds(monkeypatch):
    from firepipe import http
    calls = {"n": 0}

    class R:
        status_code = 200
        text = "ok"

    def flaky(url, params=None, timeout=None, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise http.requests.Timeout("slow")
        return R()

    monkeypatch.setattr(http.requests, "get", flaky)
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    assert http.get("https://x").text == "ok" and calls["n"] == 3


def test_find_duplicates():
    f = pd.DataFrame({"fire_id": ["A", "B", "C"], "center_lat": [26.266, 26.2716, 26.266],
                      "center_lon": [113.272, 113.273, 113.272],
                      "ignition_date": ["2022-10-17", "2022-10-17", "2022-12-01"],
                      "end_date": ["2022-10-23", "2022-10-23", "2022-12-03"]})
    d = catalog.find_duplicates(f)
    assert [(a, b) for a, b, _ in d] == [("A", "B")]     # C 同地不同时，不算重复


def test_hotspot_layers_and_burn_day_weather():
    from datetime import date

    from firepipe import fireobs
    hs = pd.DataFrame({"row": [10, 10, 40], "col": [10, 12, 40],
                       "acq_date": [date(2022, 10, 17), date(2022, 10, 18), date(2022, 10, 19)],
                       "frp": [5.0, 20.0, 50.0]})
    L = fireobs.hotspot_layers(hs, (60, 60), 30, date(2022, 10, 17), radius_m=150)
    assert L["hs_dist"][10, 10] == 0 and L["hs_dist"][10, 15] == pytest.approx(90)
    assert L["burn_day"][11, 11] in (0, 1) and L["burn_day"][45, 45] == 2
    assert L["hs_count_750"][10, 11] == 2 and L["hs_frp_max_750"][10, 11] == 20
    daily = pd.DataFrame({"tmax": [30, 31, 32], "rhmin": [20, 25, 30], "windmax": [3, 4, 5], "vpdmax": [2, 2, 3]},
                         index=pd.to_datetime(["2022-10-17", "2022-10-18", "2022-10-19"]))
    W = fireobs.burn_day_weather(L["burn_day"], daily, date(2022, 10, 17))
    assert W["bd_tmax"][45, 45] == 32
    C = fireobs.context_layers({"nbr_pre": np.ones((30, 30)), "ndvi_pre": np.ones((30, 30))}, 30)
    assert np.allclose(C["nbr_pre_mean300"][5:-5, 5:-5], 1) and np.allclose(C["ndvi_pre_sd300"][5:-5, 5:-5], 0)
