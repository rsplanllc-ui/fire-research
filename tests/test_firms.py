from datetime import date

import numpy as np
import pandas as pd

from firepipe.firms import cluster_events, events_to_fires, nearest_event


def _pts():
    rng = np.random.default_rng(1)
    a = pd.DataFrame({"latitude": 26.0 + rng.normal(0, 0.003, 30), "longitude": 113.0 + rng.normal(0, 0.003, 30),
                      "acq_date": [date(2022, 10, 17 + i % 3) for i in range(30)], "confidence": "n"})
    b = pd.DataFrame({"latitude": 26.3 + rng.normal(0, 0.003, 10), "longitude": 113.4 + rng.normal(0, 0.003, 10),
                      "acq_date": [date(2022, 11, 2)] * 10, "confidence": "h"})
    noise = pd.DataFrame({"latitude": [25.5], "longitude": [112.5], "acq_date": [date(2022, 10, 1)],
                          "confidence": ["n"]})
    low = pd.DataFrame({"latitude": [26.0] * 6, "longitude": [113.0] * 6, "acq_date": [date(2022, 12, 1)] * 6,
                        "confidence": ["l"] * 6})
    return pd.concat([a, b, noise, low], ignore_index=True)


def test_cluster_events_dates_and_filters():
    ev = cluster_events(_pts())
    assert len(ev) == 2                                   # 单个噪声点和低置信度点被去掉
    e0 = ev.iloc[0]
    assert e0["ignition_date"] == date(2022, 10, 17) and e0["end_date"] == date(2022, 10, 19)
    assert abs(e0["center_lat"] - 26.0) < 0.01
    hit = nearest_event(ev, 26.31, 113.41)
    assert hit["n_points"] == 10
    fires = events_to_fires(ev, "湖南")
    assert fires["ignition_date"].iloc[1] == date(2022, 11, 2) and fires["name_cn"].str.contains("湖南").all()


def test_same_place_different_time_splits():
    p = pd.DataFrame({"latitude": [26.0] * 12, "longitude": [113.0] * 12,
                      "acq_date": [date(2022, 10, 1)] * 6 + [date(2022, 10, 20)] * 6, "confidence": "n"})
    assert len(cluster_events(p)) == 2                    # 同地不同时 → 两场火
