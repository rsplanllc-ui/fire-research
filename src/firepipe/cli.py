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
    fires = pd.DataFrame(db.select("fires", {"select": "*", "qc_status": "neq.excluded"}))
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
    res = build.process_queue(cfg, db, args.limit)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:5000])


def cmd_train(args, cfg, db):
    dev = load_samples(db, cfg, "dev")
    out = model.run_dev(cfg, dev, args.groups, args.algos)
    for algo, res in out.items():
        eid = model.log_experiment(db, cfg, f"dev_cv_{algo}", algo, args.groups, res, "cv")
        print(algo, eid, json.dumps(res["summary"], ensure_ascii=False))


def cmd_ablation(args, cfg, db):
    dev = load_samples(db, cfg, "dev")
    print(model.ablation(cfg, dev, args.algo).to_string(index=False))


def cmd_final(args, cfg, db):
    if not args.confirm:
        sys.exit("最终测试只应在开发完成后运行一次。确认请加 --confirm")
    prev = db.select("fire_metrics", {"eval_set": "eq.test", "select": "experiment_id", "limit": "1"})
    note = "最终测试" + ("（注意：之前已运行过测试，测试集已不再是全新的）" if prev else "（首次）")
    dev, test = load_samples(db, cfg, "dev"), load_samples(db, cfg, "test")
    res, m = model.final_test(cfg, dev, test, args.algo, args.groups)
    path = f"models/{cfg['pipeline_version']}/{args.algo}_{now()[:19].replace(':', '')}.joblib"
    tmp = Path(tempfile.mkdtemp()) / "m.joblib"
    tmp.write_bytes(model.dump_model(m, feature_list(cfg, args.groups)))
    db.upload(tmp, path)
    eid = model.log_experiment(db, cfg, "final_test", args.algo, args.groups, res, "test", path, note)
    print(note, eid, json.dumps(res["summary"], ensure_ascii=False))


def _upsert_fires(db, df):
    cols = ["fire_id", "name_cn", "province", "place", "year", "ignition_date", "end_date",
            "center_lat", "center_lon", "boundary_path", "qc_status", "qc_notes", "legacy_drive_folder_id"]
    rows = df.reindex(columns=cols).astype(object).where(df.reindex(columns=cols).notna(), None)
    db.upsert("fires", rows.to_dict("records"), on_conflict="fire_id")


def cmd_discover(args, cfg, db):
    bbox = [float(v) for v in args.bbox.split(",")]
    pts = firms.fetch(bbox, date.fromisoformat(args.start), date.fromisoformat(args.end))
    ev = firms.cluster_events(pts, min_points=args.min_points)
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
        if f.get("ignition_date") and not args.all:
            continue
        lat, lon = f["center_lat"], f["center_lon"]
        if f.get("ignition_date"):
            a = date.fromisoformat(f["ignition_date"]) - timedelta(days=15)
            b = date.fromisoformat(f["end_date"] or f["ignition_date"]) + timedelta(days=15)
        elif f.get("year"):
            a, b = date(int(f["year"]), 1, 1), date(int(f["year"]), 12, 31)
        else:
            continue
        pts = firms.fetch([lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05], a, b)
        e = firms.nearest_event(firms.cluster_events(pts, min_points=3), lat, lon)
        if e is None:
            print(f["fire_id"], "附近没有找到火点")
            continue
        db.update("fires", {"fire_id": f["fire_id"]},
                  {"ignition_date": e.ignition_date, "end_date": e.end_date,
                   "qc_notes": f"起止日期由 FIRMS 火点确定（{int(e.n_points)} 个火点）",
                   "qc_status": "pending" if f["qc_status"] == "excluded" else f["qc_status"]})
        print(f["fire_id"], e.ignition_date, "→", e.end_date, int(e.n_points), "个火点")


def main(argv=None):
    p = argparse.ArgumentParser(prog="firepipe")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("import-fires"); a.add_argument("path"); a.set_defaults(f=cmd_import)
    a = sub.add_parser("discover"); a.add_argument("--bbox", required=True)
    a.add_argument("--start", required=True); a.add_argument("--end", required=True)
    a.add_argument("--province"); a.add_argument("--min-points", type=int, default=5)
    a.set_defaults(f=cmd_discover)
    a = sub.add_parser("refine-dates"); a.add_argument("--all", action="store_true",
                                                        help="已有日期的也重新核对")
    a.set_defaults(f=cmd_refine)
    a = sub.add_parser("split"); a.add_argument("--lock", action="store_true"); a.set_defaults(f=cmd_split)
    a = sub.add_parser("process"); a.add_argument("--limit", type=int, default=20); a.set_defaults(f=cmd_process)
    a = sub.add_parser("train"); a.add_argument("--groups", nargs="*"); a.add_argument("--algos", nargs="*")
    a.set_defaults(f=cmd_train)
    a = sub.add_parser("ablation"); a.add_argument("--algo", default="lightgbm"); a.set_defaults(f=cmd_ablation)
    a = sub.add_parser("final-test"); a.add_argument("--algo", required=True)
    a.add_argument("--groups", nargs="*"); a.add_argument("--confirm", action="store_true")
    a.set_defaults(f=cmd_final)
    args = p.parse_args(argv)
    args.f(args, load_config(), DB())


if __name__ == "__main__":
    main()
