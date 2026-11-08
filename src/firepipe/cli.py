"""命令行入口：python -m firepipe.cli <命令>

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

import pandas as pd

from . import build, catalog, model
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


def main(argv=None):
    p = argparse.ArgumentParser(prog="firepipe")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("import-fires"); a.add_argument("path"); a.set_defaults(f=cmd_import)
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
