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


def ablation(cfg: dict, df: pd.DataFrame, algo: str = "lightgbm", groups: list[str] | None = None) -> pd.DataFrame:
    """逐组删减因子：删掉某一组后，留出火场的逐场 R² 中位数下降多少。"""
    allg = list(groups or cfg["model"]["features"])
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


def cv_variable_importance(cfg: dict, df: pd.DataFrame, algo: str, groups: list[str] | None = None,
                           n_rep: int = 3, seed: int = 0) -> pd.DataFrame:
    """按火场分组交叉验证下的逐变量置换重要性：每折在留出火场上打乱一个变量，看逐场 R² 中位数掉多少。
    用留出火场而不是训练集，衡量的是“对新火场泛化”的贡献。"""
    feats = feature_list(cfg, groups)
    target = cfg["model"]["target"]
    X, y = _xy(df, feats, target)
    fires = df["fire_id"].values
    rng = np.random.default_rng(seed)
    preds = {f: np.full(len(df), np.nan) for f in ["__base__"] + feats}
    for tr, te in GroupKFold(n_splits=cfg["model"]["cv_folds"]).split(X, y, fires):
        m = clone(make_model(algo)).fit(X.iloc[tr], y[tr])
        Xte = X.iloc[te]
        preds["__base__"][te] = m.predict(Xte)
        for f in feats:
            acc = np.zeros(len(te))
            for _ in range(n_rep):
                Xp = Xte.copy(); Xp[f] = Xte[f].values[rng.permutation(len(te))]
                acc += m.predict(Xp)
            preds[f][te] = acc / n_rep

    def med(p):
        d = pd.DataFrame({"fire_id": fires, "y": y, "p": p})
        return float(d.groupby("fire_id").apply(lambda g: r2(g["y"].values, g["p"].values)).median())
    base = med(preds["__base__"])
    grp = {c: g for g, cols in cfg["model"]["features"].items() for c in cols}
    rows = [{"variable": f, "group": grp.get(f, "?"), "r2_median_drop": base - med(preds[f])} for f in feats]
    out = pd.DataFrame(rows).sort_values("r2_median_drop", ascending=False).reset_index(drop=True)
    out.attrs["base"] = base
    return out


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
        for k in ("r2", "r2_centered"):
            if k in r:
                r[k] = None if r[k] is None or not np.isfinite(r[k]) else r[k]
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



def aggregate_samples(df: pd.DataFrame, agg_px: int) -> pd.DataFrame:
    """把样本按 agg_px×agg_px 像元的方块聚合（数值列取均值），用于“粗尺度上还有多少可解释性”的诊断。
    row/col 改为方块左上角，便于后续再按 1 km 方块分组。"""
    if agg_px <= 1:
        return df
    d = df.copy()
    d["_br"], d["_bc"] = d["row"] // agg_px, d["col"] // agg_px
    num = d.select_dtypes("number").columns.difference(["row", "col", "_br", "_bc"])
    g = d.groupby(["fire_id", "_br", "_bc"], sort=False)
    out = g[list(num)].mean()
    out["n_agg"] = g.size()
    out = out.reset_index()
    out["row"], out["col"] = out["_br"] * agg_px, out["_bc"] * agg_px
    return out.drop(columns=["_br", "_bc"])


def within_fire_ceiling(cfg: dict, df: pd.DataFrame, groups: list[str] | None = None,
                        block_px: int = 33, algo: str = "random_forest", min_blocks: int = 5) -> pd.DataFrame:
    """诊断：同一场火内部，按约 1 km 的空间方块分组交叉验证（用本场其他方块训练）。
    反映现有因子对“场内差异”的解释上限；不是验收口径（验收要求完全没见过的火场）。"""
    feats = feature_list(cfg, groups)
    target = cfg["model"]["target"]
    rows = []
    for fid, d in df.groupby("fire_id"):
        d = d[np.isfinite(d[target])]
        blocks = (d["row"] // block_px).astype(str) + "_" + (d["col"] // block_px).astype(str)
        nb = blocks.nunique()
        if nb < min_blocks or len(d) < 50:
            continue
        X, y = _xy(d, feats, target)
        X = X.loc[:, X.notna().any()]
        pred = np.full(len(d), np.nan)
        for tr, te in GroupKFold(n_splits=min(5, nb)).split(X, y, blocks):
            m = clone(make_model(algo)).fit(X.iloc[tr], y[tr])
            pred[te] = m.predict(X.iloc[te])
        rows.append({"fire_id": fid, "n_pixels": len(d), "n_blocks": nb, "r2": r2(y, pred),
                     "dnbr_mean": float(np.mean(y)), "dnbr_std": float(np.std(y))})
    return pd.DataFrame(rows)


# ---------------- 分级 / 二分类口径（与文献的烈度潜势图对齐） ----------------
# dNBR 分级阈值参考 Key & Benson (2006)：<0.10 未烧/极低，0.10–0.27 低，0.27–0.44 中低，0.44–0.66 中高，>0.66 高
SEVERITY_BINS = [-np.inf, 0.27, 0.44, np.inf]      # 三级：低 / 中 / 高
HIGH_THRESHOLD = 0.44                                # 二分：是否中高以上


def classify_cv(cfg: dict, df: pd.DataFrame, groups: list[str] | None = None, algo: str = "lightgbm",
                folds: int | None = None, seed: int = 0) -> dict:
    """按火场分组交叉验证的分类口径：二分（是否高烈度）AUC，三级（低/中/高）总体精度与 Kappa，均按逐场和合并两种方式报告。
    用 LightGBM 分类器；其他算法用随机森林分类器。"""
    from sklearn.metrics import roc_auc_score, cohen_kappa_score, accuracy_score
    from sklearn.model_selection import GroupKFold
    feats = feature_list(cfg, groups)
    target = cfg["model"]["target"]
    d = df[np.isfinite(df[target])].reset_index(drop=True)
    X, y = _xy(d, feats, target)
    fires = d["fire_id"].values
    y_bin = (y >= HIGH_THRESHOLD).astype(int)
    y_cls = np.digitize(y, SEVERITY_BINS[1:-1])         # 0/1/2
    folds = folds or cfg["model"]["cv_folds"]

    def clf(n_classes):
        if algo == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8,
                                  colsample_bytree=0.8, random_state=seed, verbose=-1)
        from sklearn.ensemble import RandomForestClassifier
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=seed))

    p_bin = np.full(len(d), np.nan); p_cls = np.full(len(d), -1)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, fires):
        m = clf(2).fit(X.iloc[tr], y_bin[tr]); p_bin[te] = m.predict_proba(X.iloc[te])[:, 1]
        m3 = clf(3).fit(X.iloc[tr], y_cls[tr]); p_cls[te] = m3.predict(X.iloc[te])
    rows = []
    for fid in np.unique(fires):
        i = fires == fid
        auc = roc_auc_score(y_bin[i], p_bin[i]) if len(np.unique(y_bin[i])) == 2 else np.nan
        rows.append({"fire_id": fid, "n": int(i.sum()), "high_share": float(y_bin[i].mean()),
                     "auc": auc, "oa3": accuracy_score(y_cls[i], p_cls[i]),
                     "kappa3": cohen_kappa_score(y_cls[i], p_cls[i]) if len(np.unique(y_cls[i])) > 1 else np.nan})
    fm = pd.DataFrame(rows)
    summary = {"n_fires": int(len(fm)),
               "auc_pooled": float(roc_auc_score(y_bin, p_bin)),
               "auc_median_per_fire": float(fm["auc"].median()),
               "auc_q25_q75": [float(fm["auc"].quantile(.25)), float(fm["auc"].quantile(.75))],
               "share_fires_auc_ge_0.7": float((fm["auc"] >= 0.7).mean()),
               "oa3_pooled": float(accuracy_score(y_cls, p_cls)),
               "kappa3_pooled": float(cohen_kappa_score(y_cls, p_cls)),
               "oa3_median_per_fire": float(fm["oa3"].median()),
               "kappa3_median_per_fire": float(fm["kappa3"].median()),
               "high_share_overall": float(y_bin.mean()),
               "class_shares": [float((y_cls == k).mean()) for k in range(3)]}
    return {"per_fire": fm, "summary": summary}


def stratified_compare(cfg: dict, df: pd.DataFrame, strata: pd.Series, groups: list[str] | None = None,
                       algo: str = "lightgbm", min_train: int = 2000) -> dict:
    """同一套按火场分组的折里，比较"一个总模型"与"每个分层各自一个小模型"。
    strata 与 df 行对齐；分层训练样本不足 min_train 时该层退回总模型。"""
    from sklearn.metrics import roc_auc_score
    feats = feature_list(cfg, groups)
    target = cfg["model"]["target"]
    keep = np.isfinite(df[target].values)
    d = df[keep].reset_index(drop=True)
    s = np.array(["缺失" if (v is None or v != v or str(v) == "nan") else str(v) for v in np.asarray(strata, dtype=object)[keep]], dtype=object)
    X, y = _xy(d, feats, target)
    fires = d["fire_id"].values
    pg = np.full(len(d), np.nan); ps = np.full(len(d), np.nan)
    for tr, te in GroupKFold(n_splits=min(cfg["model"]["cv_folds"], len(np.unique(fires)))).split(X, y, fires):
        m = clone(make_model(algo)).fit(X.iloc[tr], y[tr])
        pg[te] = m.predict(X.iloc[te]); ps[te] = pg[te]
        for k in np.unique(s[te]):
            trk = tr[s[tr] == k]; tek = te[s[te] == k]
            if len(trk) >= min_train and len(np.unique(fires[trk])) >= 5:
                ps[tek] = clone(make_model(algo)).fit(X.iloc[trk], y[trk]).predict(X.iloc[tek])
    yb = (y >= HIGH_THRESHOLD).astype(int)

    def summ(p):
        fm = per_fire(pd.DataFrame({"fire_id": fires, "y": y, "pred": p}))
        return {"r2_median_per_fire": float(fm["r2"].median()), "r2_pooled": float(r2(y, p)),
                "auc_high_pooled": float(roc_auc_score(yb, p))}
    by = []
    for k in np.unique(s):
        i = s == k
        if i.sum() < 500 or len(np.unique(yb[i])) < 2:
            continue
        by.append({"stratum": k, "n": int(i.sum()), "fires": int(len(np.unique(fires[i]))),
                   "r2_global": float(r2(y[i], pg[i])), "r2_small": float(r2(y[i], ps[i])),
                   "auc_global": float(roc_auc_score(yb[i], pg[i])), "auc_small": float(roc_auc_score(yb[i], ps[i]))})
    return {"global": summ(pg), "small_models": summ(ps), "by_stratum": pd.DataFrame(by)}


def leave_group_out(cfg: dict, df: pd.DataFrame, group: pd.Series, groups: list[str] | None = None,
                    algo: str = "lightgbm") -> pd.DataFrame:
    """留一组验证（例如留一省份）：用其他组的火训练，在留出组的火上检验。"""
    from sklearn.metrics import roc_auc_score
    feats = feature_list(cfg, groups)
    target = cfg["model"]["target"]
    keep = np.isfinite(df[target].values)
    d = df[keep].reset_index(drop=True)
    g = np.array([str(v) for v in np.asarray(group, dtype=object)[keep]], dtype=object)
    X, y = _xy(d, feats, target)
    fires = d["fire_id"].values
    yb = (y >= HIGH_THRESHOLD).astype(int)
    rows = []
    for k in np.unique(g):
        te = g == k
        if len(np.unique(fires[te])) < 2:
            continue
        p = clone(make_model(algo)).fit(X[~te], y[~te]).predict(X[te])
        fm = per_fire(pd.DataFrame({"fire_id": fires[te], "y": y[te], "pred": p}))
        rows.append({"held_out": k, "fires": int(fm.shape[0]), "n": int(te.sum()),
                     "r2_median_per_fire": float(fm["r2"].median()), "r2_pooled": float(r2(y[te], p)),
                     "auc_high": float(roc_auc_score(yb[te], p)) if len(np.unique(yb[te])) == 2 else np.nan})
    return pd.DataFrame(rows)
