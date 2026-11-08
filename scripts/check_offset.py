"""检查 Earth Search sentinel-2-l2a 各影像的原始 DN 是否已含 +1000 偏移（对照目录里的 offset 说明）。"""
import os
import sys

import numpy as np
import odc.stac
from pystac_client import Client

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
cases = [  # (名称, bbox, 时间)
    ("江西五云镇2022秋", (114.80, 25.96, 114.86, 26.02), "2022-09-01/2022-10-16"),
    ("广西2021春", (110.00, 23.00, 110.06, 23.06), "2021-01-01/2021-02-15"),
    ("湖南2023冬", (113.20, 26.20, 113.26, 26.26), "2023-11-01/2023-12-15"),
]
c = Client.open("https://earth-search.aws.element84.com/v1")
for name, bbox, dt in cases:
    items = list(c.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=dt,
                          query={"eo:cloud_cover": {"lt": 30}}).items())[:6]
    print(f"\n== {name}：{len(items)} 景")
    for it in items:
        rb = it.assets["swir22"].extra_fields.get("raster:bands", [{}])[0]
        ds = odc.stac.load([it], bands=["red", "swir22", "scl"], bbox=bbox, crs="EPSG:4326",
                           resolution=0.0005, resampling="nearest")
        scl = ds.scl.values.ravel()
        veg = np.isin(scl, [4])
        out = []
        for b in ("red", "swir22"):
            v = ds[b].values.ravel()[veg]
            v = v[v > 0]
            out.append(f"{b} DN p1/p50={np.percentile(v,1):.0f}/{np.percentile(v,50):.0f}" if v.size else f"{b} 无植被像元")
        print(f"{it.id}  baseline={it.properties.get('s2:processing_baseline')}  "
              f"offset={rb.get('offset')}  植被像元={int(veg.sum())}  " + "  ".join(out))
    sys.stdout.flush()
