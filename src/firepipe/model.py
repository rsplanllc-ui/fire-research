"""建模与验证。

规则：
- 开发阶段只用 split='dev' 的火场，按“整场火灾”分组交叉验证（GroupKFold）。
- 封存测试（split='test'）只能通过 final_test() 运行，每次运行都会留下记录。
- 所有模型使用同一套划分和评价口径；逐火场计算 R²。
"""
from __future__ import annotations

import io
import subprocess

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import feature_list
from .metrics import per_fire, r2, summarize


def make_model(name: str, seed: int = 0):
    if name == "mean_baseline":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(n_estimators=400, min_samples_leaf=5,
                                                   max_features=0.5, n_jobs=-1, random_state=seed))
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                             min_child_samples=40, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=seed, verbose=-1)
    raise ValueError(name)


def _xy(df: pd.DataFrame, feats: list[str], target: str):
    X = df.reindex(columns=feats).astype("float32")
    return X, df[target].astype("float32").values


def grouped_cv(df: pd.DataFrame, feats: list[str], target: str, algo: str,
               folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """按火场分组交叉验证，返回每个像元的留出预测。"""
    df = df[np.isfinite(df[target])].reset_index(drop=True)
    X, y = _xy(df, feats, target)
    groups = df["fire_id"].values
    k = min(folds, len(np.unique(groups)))
    if k < 2:
        raise ValueError("开发集火场数量不足 2 个，无法按火场交叉验证")
    pred = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        m = clone(make_model(algo, seed)).fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    return pd.DataFrame({"fire_id": groups, "y": y, "pred": pred})


def run_dev(cfg: dict, df: pd.DataFrame, groups: list[str] | None = None,
            algos: list[str] | None = None) -> dict:
    """开发集：所有算法 × 指定因子组，按火场交叉验证。"""
    feats = feature_list(cfg, groups)
    target = cfg["model"]["target"]
    thr = cfg["model"]["acceptance_r2"]
    out = {}
    for a in algos or cfg["model"]["algorithms"]:
        oof = grouped_cv(df, feats, target, a, cfg["model"]["cv_folds"])
        fm = per_fire(oof)
        out[a] = {"per_fire": fm, "summary": summarize(fm, (oof["y"], oof["pred"]), thr), "oof": oof}
    return out


def ablation(cfg: dict, df: pd.DataFrame, algo: str = "lightgbm") -> pd.DataFrame:
    """逐组删减因子：删掉某一组后，留出火场的逐场 R² 中位数下降多少。"""
    allg = list(cfg["model"]["features"])
    base = run_dev(cfg, df, allg, [algo])[algo]["summary"]["r2_median"]
    rows = [{"removed": "（无）", "r2_median": base, "drop": 0.0}]
    for g in allg:
        keep = [x for x in allg if x != g]
        s = run_dev(cfg, df, keep, [algo])[algo]["summary"]["r2_median"]
        rows.append({"removed": g, "r2_median": s, "drop": base - s})
    return pd.DataFrame(rows)


def grouped_permutation(model, X: pd.DataFrame, y, cfg: dict, seed: int = 0) -> pd.DataFrame:
    """在留出数据上的置换重要性（按因子组整体打乱，减轻相关因子互相替代的问题）。"""
    rng = np.random.default_rng(seed)
    base = r2(y, model.predict(X))
    rows = []
    for g, cols in cfg["model"]["features"].items():
        cols = [c for c in cols if c in X]
        drops = []
        for _ in range(5):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            Xp[cols] = X[cols].values[idx]
            drops.append(base - r2(y, model.predict(Xp)))
        rows.append({"group": g, "r2_drop_mean": float(np.mean(drops)), "r2_drop_std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("r2_drop_mean", ascending=False)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def log_experiment(db, cfg: dict, name: str, algo: str, groups, res: dict, eval_set: str,
                   model_path: str | None = None, notes: str = "") -> int:
    exp = db.insert("experiments", {
        "name": name, "data_version": cfg["pipeline_version"], "split_version": str(cfg["split"]["seed"]),
        "features": feature_list(cfg, groups), "algorithm": algo, "params": {},
        "random_seed": 0, "code_version": git_sha(), "model_path": model_path,
        "summary": res["summary"], "notes": notes})
    rows = [{**r, "experiment_id": exp["id"], "eval_set": eval_set}
            for r in res["per_fire"].to_dict("records")]
    for r in rows:
        r["r2"] = None if not np.isfinite(r["r2"]) else r["r2"]
    db.upsert("fire_metrics", rows, on_conflict="experiment_id,fire_id,eval_set")
    return exp["id"]


def final_test(cfg: dict, dev: pd.DataFrame, test: pd.DataFrame, algo: str,
               groups: list[str] | None = None) -> tuple[dict, object]:
    """用全部开发火场训练，在封存测试火场上评价一次。"""
    feats = feature_list(cfg, groups)
    target = cfg["model"]["target"]
    dev = dev[np.isfinite(dev[target])]
    test = test[np.isfinite(test[target])]
    m = make_model(algo).fit(*_xy(dev, feats, target))
    X, y = _xy(test, feats, target)
    out = pd.DataFrame({"fire_id": test["fire_id"].values, "y": y, "pred": m.predict(X)})
    fm = per_fire(out)
    return {"per_fire": fm, "summary": summarize(fm, (out["y"], out["pred"]),
                                                 cfg["model"]["acceptance_r2"]), "pred": out}, m


def dump_model(model, feats: list[str]) -> bytes:
    buf = io.BytesIO()
    joblib.dump({"model": model, "features": feats}, buf)
    return buf.getvalue()

