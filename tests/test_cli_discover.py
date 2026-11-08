from types import SimpleNamespace

from firepipe import cli, firms
from firepipe.config import load_config
from test_firms import _pts


class FakeDB:
    def __init__(self):
        self.fires = []

    def select(self, table, params=None):
        return [dict(f) for f in self.fires] if table == "fires" else []

    def upsert(self, table, rows, on_conflict=None):
        for r in rows:
            r = {**r, "ignition_date": str(r["ignition_date"]) if r.get("ignition_date") else None}
            self.fires = [f for f in self.fires if f["fire_id"] != r["fire_id"]] + [r]
        return rows


def test_discover_inserts_and_dedups(monkeypatch):
    monkeypatch.setattr(firms, "fetch", lambda *a, **k: _pts())
    db = FakeDB()
    args = SimpleNamespace(bbox="112,25,114,27", start="2022-10-01", end="2022-12-31",
                           province="湖南", min_points=5)
    cli.cmd_discover(args, load_config(), db)
    assert len(db.fires) == 2
    assert {f["fire_id"] for f in db.fires} == {"HN_2022_001", "HN_2022_002"}
    assert all(f["qc_status"] == "pending" for f in db.fires)
    cli.cmd_discover(args, load_config(), db)      # 再跑一次不重复
    assert len(db.fires) == 2
