from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache
def load_config(path: str | None = None) -> dict:
    p = Path(path or os.environ.get("FIREPIPE_CONFIG", ROOT / "config.yaml"))
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def feature_list(cfg: dict, groups: list[str] | None = None) -> list[str]:
    feats = cfg["model"]["features"]
    groups = groups or list(feats)
    out: list[str] = []
    for g in groups:
        out.extend(feats[g])
    return out
