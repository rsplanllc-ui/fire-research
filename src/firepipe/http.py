"""带自动重试的 GET：公开数据接口偶尔超时或限流，等一会儿再试。"""
from __future__ import annotations

import time

import requests

RETRY_STATUS = {429, 500, 502, 503, 504}


def get(url: str, params=None, timeout: int = 180, tries: int = 5, wait: float = 15, **kw):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, **kw)
            if r.status_code not in RETRY_STATUS:
                return r
            last = requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
        except (requests.Timeout, requests.ConnectionError) as e:
            last = e
        time.sleep(wait * (2 ** i))
    raise last
