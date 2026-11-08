"""Supabase 读写（只用 REST 接口，不依赖 supabase-py）。

需要两个环境变量（在 GitHub 仓库的 Secrets 里设置，不写进代码）：
  SUPABASE_URL           例如 https://cgoajvzmghvvxacimcvn.supabase.co
  SUPABASE_SERVICE_KEY   后台密钥（secret key / service_role）
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

BUCKET = "fire-data"


class DB:
    def __init__(self, url: str | None = None, key: str | None = None, timeout: int = 60):
        self.url = (url or os.environ["SUPABASE_URL"]).strip().rstrip("/")   # 去掉粘贴时带进来的空格/换行
        key = (key or os.environ["SUPABASE_SERVICE_KEY"]).strip()
        self.h = {"apikey": key, "Authorization": f"Bearer {key}"}
        self.timeout = timeout

    # ---------- 表 ----------
    @staticmethod
    def _check(r):
        if not r.ok:
            raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.text[:1000]}", response=r)

    def select(self, table: str, params: dict | None = None) -> list[dict]:
        r = requests.get(f"{self.url}/rest/v1/{table}", headers=self.h,
                         params=params or {}, timeout=self.timeout)
        self._check(r)
        return r.json()

    def upsert(self, table: str, rows: list[dict] | dict, on_conflict: str | None = None) -> list[dict]:
        rows = rows if isinstance(rows, list) else [rows]
        keys = sorted({k for r in rows for k in r})          # 批量写入要求每行字段一致
        rows = [{k: r.get(k) for k in keys} for r in rows]
        h = {**self.h, "Content-Type": "application/json",
             "Prefer": "resolution=merge-duplicates,return=representation"}
        params = {"on_conflict": on_conflict} if on_conflict else {}
        r = requests.post(f"{self.url}/rest/v1/{table}", headers=h, params=params,
                          data=json.dumps(rows, default=str), timeout=self.timeout)
        self._check(r)
        return r.json()

    def insert(self, table: str, row: dict) -> dict:
        h = {**self.h, "Content-Type": "application/json", "Prefer": "return=representation"}
        r = requests.post(f"{self.url}/rest/v1/{table}", headers=h,
                          data=json.dumps(row, default=str), timeout=self.timeout)
        self._check(r)
        return r.json()[0]

    def update(self, table: str, match: dict, values: dict) -> list[dict]:
        h = {**self.h, "Content-Type": "application/json", "Prefer": "return=representation"}
        params = {k: f"eq.{v}" for k, v in match.items()}
        r = requests.patch(f"{self.url}/rest/v1/{table}", headers=h, params=params,
                           data=json.dumps(values, default=str), timeout=self.timeout)
        self._check(r)
        return r.json()

    # ---------- 文件 ----------
    def upload(self, local: str | Path, remote: str, content_type: str = "application/octet-stream") -> str:
        data = Path(local).read_bytes()
        h = {**self.h, "Content-Type": content_type, "x-upsert": "true"}
        r = requests.post(f"{self.url}/storage/v1/object/{BUCKET}/{remote}", headers=h,
                          data=data, timeout=max(self.timeout, 300))
        self._check(r)
        return remote

    def download(self, remote: str, local: str | Path) -> Path:
        r = requests.get(f"{self.url}/storage/v1/object/{BUCKET}/{remote}", headers=self.h,
                         timeout=max(self.timeout, 300))
        self._check(r)
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(r.content)
        return Path(local)

    # ---------- 运行日志 ----------
    def run_start(self, job: str, fire_id: str | None = None) -> int:
        row = self.insert("pipeline_runs", {"job": job, "fire_id": fire_id, "status": "running",
                                            "started_at": now()})
        return row["id"]

    def run_end(self, run_id: int, status: str, log: str = "") -> None:
        self.update("pipeline_runs", {"id": run_id},
                    {"status": status, "finished_at": now(), "log": log[-20000:]})


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
