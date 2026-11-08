"""火场名单：导入、编号、封存测试集划分。"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

PROVINCE_CODE = {
    "湖南": "HN", "江西": "JX", "广东": "GD", "广西": "GX", "福建": "FJ",
    "浙江": "ZJ", "安徽": "AH", "湖北": "HB", "贵州": "GZ", "云南": "YN",
    "四川": "SC", "重庆": "CQ", "海南": "HI",
}

REQUIRED = ["name_cn", "ignition_date", "end_date", "center_lat", "center_lon"]


def parse_name(name: str) -> dict:
    """从文件夹名里解析年份和省份，例如 '2022湖南衡阳祁东县3'。"""
    year = None
    m = re.match(r"^(\d{4})", name)
    if m:
        year = int(m.group(1))
    prov = next((p for p in PROVINCE_CODE if p in name), None)
    return {"year": year, "province": prov}


def make_ids(df: pd.DataFrame) -> pd.Series:
    """按 省份代码_年份_序号 生成稳定编号（按名称排序，保证可复现）。"""
    df = df.copy()
    df["_code"] = df["province"].map(PROVINCE_CODE).fillna("XX")
    df["_year"] = df["year"].fillna(0).astype(int)
    df = df.sort_values(["_code", "_year", "name_cn"])
    df["_seq"] = df.groupby(["_code", "_year"]).cumcount() + 1
    ids = df["_code"] + "_" + df["_year"].astype(str) + "_" + df["_seq"].map("{:03d}".format)
    return ids.reindex(df.index)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """整理用户提供的火场清单（CSV/Excel），返回可写入 fires 表的行。"""
    df = df.copy()
    missing = [c for c in ["name_cn"] if c not in df]
    if missing:
        raise ValueError(f"清单缺少列：{missing}")
    parsed = df["name_cn"].map(parse_name).apply(pd.Series)
    for c in ["year", "province"]:
        if c not in df or df[c].isna().all():
            df[c] = parsed[c]
        else:
            df[c] = df[c].fillna(parsed[c])
    for c in ["ignition_date", "end_date"]:
        if c in df:
            df[c] = pd.to_datetime(df[c].astype(str), errors="coerce", format="mixed").dt.date
    if "fire_id" not in df or df["fire_id"].isna().any():
        df["fire_id"] = make_ids(df)
    complete = df[[c for c in REQUIRED if c in df]].notna().all(axis=1) & all(c in df for c in REQUIRED)
    df["qc_status"] = np.where(complete, "pending", "excluded")
    df["qc_notes"] = np.where(complete, None, "缺少日期或坐标，暂不处理")
    return df


def stratified_split(fires: pd.DataFrame, test_fraction: float, by: list[str], seed: int) -> pd.Series:
    """按分层随机抽取封存测试火场。每层至少保留 1 个开发火场。"""
    rng = np.random.default_rng(seed)
    split = pd.Series("dev", index=fires.index)
    keys = fires[by].astype(str).agg("|".join, axis=1) if by else pd.Series("all", index=fires.index)
    for _, idx in fires.groupby(keys).groups.items():
        idx = sorted(idx, key=lambda i: fires.loc[i, "fire_id"])
        n = len(idx)
        k = int(round(n * test_fraction))
        k = min(k, n - 1) if n > 1 else 0
        if k > 0:
            pick = rng.choice(idx, size=k, replace=False)
            split.loc[pick] = "test"
    return split
