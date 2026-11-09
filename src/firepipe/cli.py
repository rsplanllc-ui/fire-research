"""命令行入口：python -m firepipe.cli <命令>

  discover --bbox W,S,E,N --start D --end D [--province 湖南] [--min-points 5]
                                  用 VIIRS 火点自动发现火灾并写入名单
  refine-dates                    用 VIIRS 火点给“有坐标、缺日期/日期待核对”的火场补起止日期
  import-fires  清单.csv          导入/更新火场名单
  split [--lock]                  按分层随机划分封存测试火场（--lock 后不可更改）
  process [--limit N]             处理所有待处理火场（取数、质检、采样、入库）
  train [--groups ...]            开发集按火场交叉验证，结果写入 experiments / fire_metrics
  ablation                        删减因子组实验
  final-test --algo A --confirm   用封存测试火场做一次最终评价（会留下永久记录）
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from datetime import date, timedelta

from . import build, catalog, firms, model
from .config import feature_list, load_config
from .db import DB, now


def load_samples(db: DB, cfg: dict, split: str) -> pd.DataFrame:
    ver = cfg["pipeline_version"]
    fires = {f["fire_id"]: f for f in db.select("fires", {"select": "fire_id,split,qc_status"})}
    assets = db.select("data_assets", {"variable": "eq.samples", "pipeline_version": f"eq.{ver}",
                                       "select": "fire_id,storage_path"})
    tmp = Path(tempfile.mkdtemp())
    parts = []
    for a in assets:
        f = fires.get(a["fire_id"])
        if not f or f["split"] != split or f["qc_status"] != "ok":
            continue
        parts.append(pd.read_parquet(db.download(a["storage_path"], tmp / f"{a['fire_id']}.parquet")))
    if not parts:
        raise SystemExit(f"没有 split={split} 且质检通过的样本")
    return pd.concat(parts, ignore_index=True)


def cmd_import(args, cfg, db):
    src = Path(args.path)
    df = pd.read_excel(src) if src.suffix in {".xlsx", ".xls"} else pd.read_csv(src)
    df = catalog.prepare(df)
    cols = ["fire_id", "name_cn", "province", "place", "year", "ignition_date", "end_date",
            "center_lat", "center_lon", "boundary_path", "qc_status", "qc_notes", "legacy_drive_folder_id"]
    rows = df.reindex(columns=cols).astype(object).where(df.reindex(columns=cols).notna(), None)
    db.upsert("fires", rows.to_dict("records"), on_conflict="fire_id")
    print(f"导入 {len(rows)} 个火场；资料齐全 {int((df.qc_status == 'pending').sum())} 个")


def cmd_split(args, cfg, db):
    fires = pd.DataFrame(db.select("fires", {"select": "*", "qc_status": "eq.ok"}))   # 只给质检通过的火场划分
    if fires["split_locked_at"].notna().any():
        print("已有锁定的划分，只给新火场补充划分")
        fires = fires[fires["split_locked_at"].isna()]
    s = catalog.stratified_split(fires, cfg["split"]["test_fraction"], cfg["split"]["stratify_by"],
                                 cfg["split"]["seed"])
    for fid, sp in zip(fires["fire_id"], s):
        vals = {"split": sp}
        if args.lock:
            vals["split_locked_at"] = now()
        db.update("fires", {"fire_id": fid}, vals)
    print(s.value_counts().to_dict(), "已锁定" if args.lock else "（未锁定，可重新划分）")


def cmd_process(args, cfg, db):
    res = build.process_queue(cfg, db, args.limit, args.shard, args.nshards, args.split or None)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:5000])


def _groups(args, cfg):
    if getattr(args, "scenario", None):
        return cfg["model"]["scenarios"][args.scenario]
    return args.groups


def cmd_train(args, cfg, db):
    dev = load_samples(db, cfg, "dev")
    groups = _groups(args, cfg)
    tag = args.scenario or "custom"
    print(f"开发集：{dev.fire_id.nunique()} 场火，{len(dev)} 个样本；情景 {tag}：{groups}")
    out = model.run_dev(cfg, dev, groups, args.algos)
    for algo, res in out.items():
        eid = model.log_experiment(db, cfg, f"dev_cv_{tag}_{algo}", algo, groups, res, "cv")
        print(algo, eid, json.dumps(res["summary"], ensure_ascii=False))


def cmd_ablation(args, cfg, db):
    dev = load_samples(db, cfg, "dev")
    print(model.ablation(cfg, dev, args.algo, _groups(args, cfg)).to_string(index=False))


def cmd_diag(args, cfg, db):
    dev = load_samples(db, cfg, "dev")
    feats = feature_list(cfg) + [cfg["model"]["target"]]
    miss = dev.groupby("fire_id")[[f for f in feats if f in dev]].apply(lambda d: d.isna().mean()).round(2)
    bad = miss.loc[:, (miss > 0.2).any()]
    print("缺失比例 >20% 的因子（按火场）：")
    print(bad[(bad > 0.2).any(axis=1)].to_string())
    print(dev[[f for f in feats if f in dev]].describe().T.round(3).to_string())


def cmd_within(args, cfg, db):
    dev = load_samples(db, cfg, "dev")
    res_m = cfg["grid"]["resolution_m"]
    scales = [int(v) for v in (args.scales or "30").split(",")]
    for sc in scales:
        agg_px = max(1, int(round(sc / res_m)))
        d = model.aggregate_samples(dev, agg_px)
        res = model.within_fire_ceiling(cfg, d, _groups(args, cfg), min_blocks=4)
        if len(scales) == 1:
            fires = pd.DataFrame(db.select("fires", {"select": "fire_id,ignition_date,province"}))
            res = res.merge(fires, on="fire_id", how="left")
            res["year"] = pd.to_datetime(res["ignition_date"]).dt.year
            print(res.drop(columns=["ignition_date"]).round(3).to_string(index=False))
            res["sev"] = pd.cut(res["dnbr_mean"], [-1, 0.25, 0.4, 9], labels=["低(<0.25)", "中(0.25-0.4)", "高(>0.4)"])
            res["var"] = pd.cut(res["dnbr_std"], [-1, 0.15, 0.22, 9], labels=["小(<0.15)", "中(0.15-0.22)", "大(>0.22)"])
            for key, title in [("sev", "按火场平均 dNBR"), ("var", "按场内 dNBR 标准差"), ("year", "按年份"), ("province", "按省份")]:
                g = res.groupby(key, observed=True)["r2"].agg(["median", "count"]).round(3)
                print(f"\n{title}分组的场内 R² 中位数：\n{g.to_string()}")
        print(f"[聚合尺度 {agg_px * res_m:.0f} m，{len(d)} 个样本] 同场内部（1 km 方块交叉验证）逐场 R² 中位数：",
              round(float(res["r2"].median()), 3), "；四分位：", res["r2"].quantile([0.25, 0.75]).round(3).tolist(),
              "；火场数：", len(res))


def cmd_cnn(args, cfg, db):
    """CNN 对比实验：按火场分组交叉验证，和树模型同口径。"""
    from . import cnn
    dev = load_samples(db, cfg, "dev")
    feats = model.feature_list(cfg, _groups(args, cfg))
    if args.max_per_fire:
        dev = dev.sample(frac=1, random_state=0)
        dev = dev[dev.groupby("fire_id").cumcount() < args.max_per_fire].reset_index(drop=True)
    print(f"开发集 {dev['fire_id'].nunique()} 场火，{len(dev)} 个样本；通道 {len(feats)}+1；图块 {args.patch}×{args.patch}")
    X, y, fids, chans = cnn.build_patches(db, cfg, dev, feats, patch=args.patch)
    print(f"图块张量 {X.shape}，{X.nbytes / 1e9:.2f} GB")
    res = cnn.run_cnn_cv(cfg, X, y, fids, epochs=args.epochs)
    print(json.dumps(res["summary"], ensure_ascii=False))
    scen = args.scenario or "custom"
    eid = model.log_experiment(db, cfg, f"dev_cv_cnn_{scen}", "cnn", _groups(args, cfg), res, "cv",
                               notes=f"patch={args.patch} epochs={args.epochs}")
    print("实验编号", eid)


def cmd_importance(args, cfg, db):
    dev = load_samples(db, cfg, "dev")
    res = model.cv_variable_importance(cfg, dev, args.algo, _groups(args, cfg))
    print(f"基线逐场 R² 中位数 {res.attrs['base']:.3f}；逐变量置换后下降（按火场交叉验证的留出预测）：")
    print(res.round(4).to_string(index=False))
    print("\n按因子组汇总（组内变量下降之和）：")
    print(res.groupby("group")["r2_median_drop"].sum().sort_values(ascending=False).round(4).to_string())


def cmd_inspect(args, cfg, db):
    """抽查出图：每场火 4 幅——火前 NBR、去背景 dNBR、火烧范围+火点、土地覆盖。输出到 inspect/ 目录。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rioxarray  # noqa: F401
    import rasterio
    ver = cfg["pipeline_version"]
    fires = {f["fire_id"]: f for f in db.select("fires", {"select": "fire_id,name_cn,province,ignition_date,split,qc_status"})}
    ids = args.fires or sorted(f for f, v in fires.items() if v["split"] == "dev" and v["qc_status"] == "ok")[:args.n]
    out = Path("inspect"); out.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    for fid in ids:
        L = {}
        for k in ("nbr_pre", "dnbr_offset", "perimeter", "hs_dist", "worldcover", "burn_day"):
            try:
                with rasterio.open(db.download(f"processed/{ver}/{fid}/{k}.tif", tmp / f"{fid}_{k}.tif")) as src:
                    L[k] = src.read(1)
            except Exception as e:
                print(f"{fid} 缺 {k}：{e}")
        if "dnbr_offset" not in L:
            continue
        f = fires[fid]
        fig, ax = plt.subplots(1, 4, figsize=(20, 5.2))
        fig.suptitle(f"{fid} {f.get('name_cn') or ''} {f.get('province') or ''} {f.get('ignition_date')}", fontsize=13)
        im = ax[0].imshow(L.get("nbr_pre"), cmap="RdYlGn", vmin=-0.2, vmax=0.8); ax[0].set_title("NBR pre-fire"); plt.colorbar(im, ax=ax[0], fraction=0.04)
        im = ax[1].imshow(L["dnbr_offset"], cmap="inferno_r", vmin=-0.2, vmax=0.8); ax[1].set_title("dNBR (background removed)"); plt.colorbar(im, ax=ax[1], fraction=0.04)
        ax[2].imshow(L["dnbr_offset"], cmap="Greys_r", vmin=-0.3, vmax=1.0)
        if "perimeter" in L:
            ax[2].contour(L["perimeter"] > 0.5, levels=[0.5], colors="red", linewidths=0.8)
        if "hs_dist" in L:
            rr, cc = np.where(L["hs_dist"] <= 0.5)
            bd = L.get("burn_day")
            ax[2].scatter(cc, rr, s=6, c=(bd[rr, cc] if bd is not None else "cyan"), cmap="viridis", edgecolors="none")
        ax[2].set_title("perimeter (red) + hotspots (color=burn day)")
        if "worldcover" in L:
            ax[3].imshow(L["worldcover"], cmap="tab20", vmin=10, vmax=100); ax[3].set_title("WorldCover (10=tree)")
            if "perimeter" in L:
                ax[3].contour(L["perimeter"] > 0.5, levels=[0.5], colors="red", linewidths=0.8)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.tight_layout()
        fig.savefig(out / f"{fid}.png", dpi=110)
        plt.close(fig)
        inside = L["dnbr_offset"][L.get("perimeter", np.ones_like(L["dnbr_offset"])) > 0.5]
        print(f"{fid}: 范围内 dNBR 均值 {np.nanmean(inside):.3f} 标准差 {np.nanstd(inside):.3f}，"
              f"<0.1 的像元占 {np.nanmean(inside < 0.1):.2f}，>0.44 的占 {np.nanmean(inside > 0.44):.2f}")


def cmd_final(args, cfg, db):
    if not args.confirm:
        sys.exit("最终测试只应在开发完成后运行一次。确认请加 --confirm")
    prev = db.select("fire_metrics", {"eval_set": "eq.test", "select": "experiment_id", "limit": "1"})
    note = "最终测试" + ("（注意：之前已运行过测试，测试集已不再是全新的）" if prev else "（首次）")
    dev, test = load_samples(db, cfg, "dev"), load_samples(db, cfg, "test")
    groups = _groups(args, cfg)
    res, m = model.final_test(cfg, dev, test, args.algo, groups)
    path = f"models/{cfg['pipeline_version']}/{args.algo}_{now()[:19].replace(':', '')}.joblib"
    tmp = Path(tempfile.mkdtemp()) / "m.joblib"
    tmp.write_bytes(model.dump_model(m, feature_list(cfg, groups)))
    db.upload(tmp, path)
    eid = model.log_experiment(db, cfg, f"final_test_{args.scenario or 'custom'}", args.algo, groups, res,
                               "test", path, note)
    print(note, eid, json.dumps(res["summary"], ensure_ascii=False))


def _upsert_fires(db, df):
    cols = ["fire_id", "name_cn", "province", "place", "year", "ignition_date", "end_date",
            "center_lat", "center_lon", "boundary_path", "qc_status", "qc_notes", "legacy_drive_folder_id"]
    rows = df.reindex(columns=cols).astype(object).where(df.reindex(columns=cols).notna(), None)
    db.upsert("fires", rows.to_dict("records"), on_conflict="fire_id")


def cmd_discover(args, cfg, db):
    bbox = [float(v) for v in args.bbox.split(",")] if isinstance(args.bbox, str) else list(args.bbox)
    pts = firms.fetch(bbox, date.fromisoformat(args.start), date.fromisoformat(args.end),
                      sources=["VIIRS_SNPP_SP"])        # 大范围查询只用一颗卫星，节省额度
    ev = firms.cluster_events(pts, min_points=args.min_points)
    if getattr(args, "top", None) and len(ev) > args.top:
        ev = ev.sort_values("n_points", ascending=False).head(args.top).sort_values("ignition_date")
    print(f"火点 {len(pts)} 个，聚成 {len(ev)} 场火灾（≥{args.min_points} 个火点）")
    if ev.empty:
        return
    new = firms.events_to_fires(ev, args.province)
    have = pd.DataFrame(db.select("fires", {"select": "fire_id,center_lat,center_lon,ignition_date"}))
    if not have.empty:   # 已在名单里的（3 km、5 天内）不重复加入
        have["ignition_date"] = pd.to_datetime(have["ignition_date"]).dt.date
        dup = []
        for i, r in new.iterrows():
            d = np.hypot(have.center_lat - r.center_lat, have.center_lon - r.center_lon) < 0.03
            t = have.ignition_date.map(lambda x: abs((x - r.ignition_date).days) <= 5 if x else False)
            dup.append(bool((d & t).any()))
        new = new[~np.array(dup)]
    if new.empty:
        print("全部已在名单里")
        return
    new = catalog.prepare(new)
    existing_ids = {f["fire_id"] for f in db.select("fires", {"select": "fire_id"})}
    new["fire_id"] = _next_ids(new, existing_ids)
    _upsert_fires(db, new)
    print(new[["fire_id", "name_cn", "ignition_date", "end_date"]].to_string(index=False))


def cmd_discover_batch(args, cfg, db):
    """按 config.yaml 的 discovery 段：每个省份 × 每个火季，各取火点最多的前 N 场。"""
    from types import SimpleNamespace
    d = cfg["discovery"]
    for prov, bbox in d["regions"].items():
        if args.provinces and prov not in args.provinces:
            continue
        seasons = [x.split(":") for x in args.seasons] if args.seasons else d["seasons"]
        for start, end in seasons:
            print(f"== {prov} {start}～{end}")
            try:
                cmd_discover(SimpleNamespace(bbox=bbox, start=start, end=end, province=prov,
                                             min_points=d["min_points"],
                                             top=args.top or d["top_per_group"]), cfg, db)
            except Exception as e:
                print("失败：", e)


def _next_ids(df, existing):
    ids = []
    for _, r in df.iterrows():
        code = catalog.PROVINCE_CODE.get(r.get("province"), "XX")
        k = 1
        while f"{code}_{int(r['year'])}_{k:03d}" in existing:
            k += 1
        fid = f"{code}_{int(r['year'])}_{k:03d}"
        existing.add(fid)
        ids.append(fid)
    return ids


def cmd_refine(args, cfg, db):
    fires = db.select("fires", {"select": "*", "center_lat": "not.is.null"})
    for f in fires:
        if args.drive:        # 只核准来自 Drive 清单、还没用火点核准过的火场
            if not f.get("legacy_drive_folder_id") or "FIRMS" in (f.get("qc_notes") or ""):
                continue
        elif f.get("ignition_date") and not args.all:
            continue
        lat, lon = f["center_lat"], f["center_lon"]
        if f.get("ignition_date"):
            a = date.fromisoformat(f["ignition_date"]) - timedelta(days=15)
            b = date.fromisoformat(f["end_date"] or f["ignition_date"]) + timedelta(days=15)
        elif f.get("year"):
            a, b = date(int(f["year"]), 1, 1), date(int(f["year"]), 12, 31)
        else:
            continue
        pts = firms.fetch([lon - 0.08, lat - 0.08, lon + 0.08, lat + 0.08], a, b)
        e = firms.nearest_event(firms.cluster_events(pts, min_points=3), lat, lon)
        if e is None:
            print(f["fire_id"], "附近没有找到火点")
            continue
        db.update("fires", {"fire_id": f["fire_id"]},
                  {"ignition_date": e.ignition_date, "end_date": e.end_date,
                   "center_lat": round(float(e.center_lat), 5), "center_lon": round(float(e.center_lon), 5),
                   "qc_notes": f"起止日期和中心由 FIRMS 火点确定（{int(e.n_points)} 个火点）；原坐标 {lat:.4f},{lon:.4f}",
                   "qc_status": "pending" if f["qc_status"] == "excluded" else f["qc_status"]})
        print(f["fire_id"], e.ignition_date, "→", e.end_date, int(e.n_points), "个火点")


def main(argv=None):
    p = argparse.ArgumentParser(prog="firepipe")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("import-fires"); a.add_argument("path"); a.set_defaults(f=cmd_import)
    a = sub.add_parser("discover"); a.add_argument("--bbox", required=True)
    a.add_argument("--start", required=True); a.add_argument("--end", required=True)
    a.add_argument("--province"); a.add_argument("--min-points", type=int, default=5)
    a.add_argument("--top", type=int)
    a.set_defaults(f=cmd_discover)
    a = sub.add_parser("discover-batch"); a.add_argument("--provinces", nargs="*")
    a.add_argument("--top", type=int, default=0, help="每省每季取前几场，0 用配置值")
    a.add_argument("--seasons", nargs="*", help="形如 2019-10-01:2020-04-30")
    a.set_defaults(f=cmd_discover_batch)
    a = sub.add_parser("refine-dates"); a.add_argument("--all", action="store_true",
                                                        help="已有日期的也重新核对")
    a.add_argument("--drive", action="store_true", help="只核准 Drive 清单导入、未核准过的火场")
    a.set_defaults(f=cmd_refine)
    a = sub.add_parser("split"); a.add_argument("--lock", action="store_true"); a.set_defaults(f=cmd_split)
    a = sub.add_parser("process"); a.add_argument("--limit", type=int, default=20)
    a.add_argument("--shard", type=int, default=0); a.add_argument("--nshards", type=int, default=1)
    a.add_argument("--split", default="", help="只处理某个划分（dev / test），留空为全部")
    a.set_defaults(f=cmd_process)
    a = sub.add_parser("train"); a.add_argument("--groups", nargs="*"); a.add_argument("--algos", nargs="*")
    a.add_argument("--scenario")
    a.set_defaults(f=cmd_train)
    a = sub.add_parser("ablation"); a.add_argument("--algo", default="lightgbm"); a.add_argument("--scenario")
    a.add_argument("--groups", nargs="*"); a.set_defaults(f=cmd_ablation)
    a = sub.add_parser("diag"); a.set_defaults(f=cmd_diag)
    a = sub.add_parser("cnn"); a.add_argument("--scenario"); a.add_argument("--groups", nargs="*")
    a.add_argument("--patch", type=int, default=15); a.add_argument("--epochs", type=int, default=12)
    a.add_argument("--max-per-fire", type=int, default=0, help="每场火最多用多少样本（省内存/时间）")
    a.set_defaults(f=cmd_cnn)
    a = sub.add_parser("importance"); a.add_argument("--algo", default="lightgbm"); a.add_argument("--scenario")
    a.add_argument("--groups", nargs="*"); a.set_defaults(f=cmd_importance)
    a = sub.add_parser("inspect"); a.add_argument("--fires", nargs="*"); a.add_argument("--n", type=int, default=8)
    a.set_defaults(f=cmd_inspect)
    a = sub.add_parser("within-fire"); a.add_argument("--scenario"); a.add_argument("--groups", nargs="*")
    a.add_argument("--scales", default="", help="聚合尺度（米），逗号分隔，如 30,90,150,300")
    a.set_defaults(f=cmd_within)
    a = sub.add_parser("final-test"); a.add_argument("--algo", required=True)
    a.add_argument("--groups", nargs="*"); a.add_argument("--scenario"); a.add_argument("--confirm", action="store_true")
    a.set_defaults(f=cmd_final)
    args = p.parse_args(argv)
    args.f(args, load_config(), DB())


if __name__ == "__main__":
    main()
