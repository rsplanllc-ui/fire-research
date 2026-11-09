"""卷积网络（CNN）对比实验：以采样像元为中心取 P×P 图块，所有因子图层叠成通道，回归去背景 dNBR。

与树模型的区别：树模型一个像元一行，只看该像元自己的因子值；CNN 能看到邻域的纹理和空间格局
（林地斑块边缘、沟谷走向、火前影像纹理）。评价口径完全一致：按火场分组交叉验证，逐场 R²。

依赖 torch（CPU 版即可）。图块在 Actions 上即时从 COG 栅格构建，不落地到数据库。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import per_fire, r2, summarize
from .model import feature_list


# ---------------- 数据：从栅格切图块 ----------------
def _read(db, ver: str, fid: str, k: str, tmp: Path) -> np.ndarray | None:
    import rasterio
    try:
        p = db.download(f"processed/{ver}/{fid}/{k}.tif", tmp / f"{fid}_{k}.tif")
        with rasterio.open(p) as src:
            return src.read(1).astype("float32")
    except Exception:
        return None


def build_patches(db, cfg: dict, samples: pd.DataFrame, feats: list[str], patch: int = 15,
                  cache_dir: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """返回 X[N, C, P, P]（float16）、y[N]、fire_id[N]、通道名。
    连续因子直接作通道；worldcover 变成“是否林地”0/1。缺失用 0 填（标准化之后），并附一个有效掩膜通道。"""
    ver = cfg["pipeline_version"]
    target = cfg["model"]["target"]
    tmp = Path(cache_dir or tempfile.mkdtemp())
    half = patch // 2
    chans = [f for f in feats]
    # 全局标准化参数：用采样表的均值/标准差（和栅格同分布）
    mu, sd = {}, {}
    for f in chans:
        col = samples[f] if f in samples else pd.Series(dtype="float32")
        if f == "worldcover":
            mu[f], sd[f] = 0.0, 1.0
        else:
            mu[f] = float(np.nanmean(col)) if col.notna().any() else 0.0
            sd[f] = float(np.nanstd(col)) if col.notna().any() and np.nanstd(col) > 0 else 1.0
    Xs, ys, fids = [], [], []
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=12)
    n_f = samples["fire_id"].nunique()
    for i_f, (fid, d) in enumerate(samples.groupby("fire_id")):
        layers = {}
        arrs = list(pool.map(lambda f: _read(db, ver, fid, f, tmp), chans))   # 并行下载该火场全部图层
        for f, a in zip(chans, arrs):
            if a is None:
                continue
            if f == "worldcover":
                a = (a == 10).astype("float32")
            else:
                a = (a - mu[f]) / sd[f]
            layers[f] = a
        if i_f % 10 == 0:
            print(f"  切图块 {i_f + 1}/{n_f} 场…", flush=True)
        if not layers:
            continue
        shape = next(iter(layers.values())).shape
        stack = np.zeros((len(chans) + 1, shape[0], shape[1]), "float32")
        for i, f in enumerate(chans):
            if f in layers:
                v = layers[f]
                stack[i] = np.nan_to_num(v, nan=0.0)
        stack[-1] = np.isfinite(layers.get(chans[0], np.zeros(shape))).astype("float32")   # 有效掩膜
        pad = np.pad(stack, ((0, 0), (half, half), (half, half)), mode="reflect")
        rows = d["row"].to_numpy(int) + half
        cols = d["col"].to_numpy(int) + half
        ok = np.isfinite(d[target].to_numpy())
        for r, c, yv, o in zip(rows, cols, d[target].to_numpy("float32"), ok):
            if not o:
                continue
            Xs.append(pad[:, r - half:r + half + 1, c - half:c + half + 1].astype("float16"))
            ys.append(yv); fids.append(fid)
    X = np.stack(Xs) if Xs else np.zeros((0, len(chans) + 1, patch, patch), "float16")
    return X, np.asarray(ys, "float32"), np.asarray(fids), chans + ["valid_mask"]


# ---------------- 模型 ----------------
def _make_net(n_ch: int, patch: int):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(n_ch, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
            self.head = nn.Sequential(nn.Linear(64 + n_ch, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
            self.c = patch // 2

        def forward(self, x):
            g = self.conv(x).flatten(1)
            center = x[:, :, self.c, self.c]          # 中心像元自身的因子值直接接到头部
            return self.head(torch.cat([g, center], 1)).squeeze(1)

    return Net()


def train_cnn(X: np.ndarray, y: np.ndarray, Xval: np.ndarray | None = None, yval: np.ndarray | None = None,
              epochs: int = 12, batch: int = 256, lr: float = 1e-3, seed: int = 0, verbose: bool = False):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    net = _make_net(X.shape[1], X.shape[2])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt = torch.from_numpy(X.astype("float32")); yt = torch.from_numpy(y)
    n = len(Xt)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xt[idx]
            # 数据增强：随机翻转/旋转（因子图层对方向不敏感的部分；坡向类已是角度差，翻转不改变其意义不严格成立，
            # 但经验上有助于泛化）
            if np.random.rand() < 0.5:
                xb = torch.flip(xb, dims=[3])
            k = np.random.randint(4)
            if k:
                xb = torch.rot90(xb, k, dims=[2, 3])
            opt.zero_grad()
            loss = torch.nn.functional.smooth_l1_loss(net(xb), yt[idx], beta=0.1)
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if verbose:
            msg = f"epoch {ep + 1}/{epochs} loss {tot / n:.4f}"
            if Xval is not None:
                msg += f" val_r2 {r2(yval, predict_cnn(net, Xval)):.3f}"
            print(msg, flush=True)
    return net


def predict_cnn(net, X: np.ndarray, batch: int = 1024) -> np.ndarray:
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(net(torch.from_numpy(X[i:i + batch].astype("float32"))).numpy())
    return np.concatenate(out) if out else np.zeros(0, "float32")


# ---------------- 评价：按火场分组交叉验证 ----------------
def run_cnn_cv(cfg: dict, X: np.ndarray, y: np.ndarray, fids: np.ndarray, epochs: int = 12,
               folds: int | None = None, verbose: bool = True, batch: int = 256) -> dict:
    from sklearn.model_selection import GroupKFold
    folds = folds or cfg["model"]["cv_folds"]
    pred = np.full(len(y), np.nan, "float32")
    for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(X, y, fids)):
        if verbose:
            print(f"-- 折 {k + 1}/{folds}：训练 {len(tr)}，留出 {len(te)}（{len(set(fids[te]))} 场火）", flush=True)
        net = train_cnn(X[tr], y[tr], X[te], y[te], epochs=epochs, batch=batch, seed=k, verbose=verbose)
        pred[te] = predict_cnn(net, X[te])
    oof = pd.DataFrame({"fire_id": fids, "y": y, "pred": pred})
    fm = per_fire(oof)
    return {"per_fire": fm, "summary": summarize(fm, (oof["y"], oof["pred"]), cfg["model"]["acceptance_r2"]), "oof": oof}
