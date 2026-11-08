"""评价指标：逐火场计算，R² 用预测决定系数（可以为负），不用相关系数平方代替。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - np.sum((y - p) ** 2) / ss_tot)


def per_fire(df: pd.DataFrame, y: str = "y", p: str = "pred", g: str = "fire_id") -> pd.DataFrame:
    rows = []
    for fid, d in df.groupby(g):
        e = d[p] - d[y]
        yc, pc = d[y] - d[y].mean(), d[p] - d[p].mean()
        rows.append({"fire_id": fid, "n_pixels": len(d), "r2": r2(d[y], d[p]),
                     "r2_centered": r2(yc, pc),     # 诊断：去掉整体偏差后，只看场内格局
                     "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
                     "bias": float(np.mean(e)), "obs_var": float(np.var(d[y]))})
    return pd.DataFrame(rows)


def summarize(fm: pd.DataFrame, pooled: tuple | None = None, threshold: float = 0.90) -> dict:
    s = {"n_fires": int(len(fm)),
         "r2_median": float(fm["r2"].median()),
         "r2_mean": float(fm["r2"].mean()),
         "r2_min": float(fm["r2"].min()),
         "share_fires_ge_threshold": float((fm["r2"] >= threshold).mean()),
         "all_fires_ge_threshold": bool((fm["r2"] >= threshold).all()),
         "rmse_median": float(fm["rmse"].median())}
    if "r2_centered" in fm:
        s["r2_centered_median"] = float(fm["r2_centered"].median())
    if pooled is not None:
        s["r2_pooled_pixels"] = r2(*pooled)   # 只作补充，不作验收依据
    return s
