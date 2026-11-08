"""火烧范围与 dNBR 偏移校正。"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def derive_perimeter(dnbr: np.ndarray, center_rc: tuple[int, int], threshold: float,
                     min_patch: int) -> np.ndarray:
    """没有官方边界时：dNBR > 阈值（不取绝对值），去掉小斑块，保留离中心点最近的连通区。
    得到的是“自动提取范围”，需要人工质检。"""
    m = np.nan_to_num(dnbr, nan=-9) > threshold
    m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_closing(m, iterations=2)
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_patch]
    if not keep:
        return np.zeros_like(m)
    r0, c0 = center_rc
    if 0 <= r0 < m.shape[0] and 0 <= c0 < m.shape[1] and lab[r0, c0] in keep:
        target = lab[r0, c0]
    else:
        coms = ndimage.center_of_mass(m, lab, keep)
        d = [np.hypot(r - r0, c - c0) for r, c in coms]
        target = keep[int(np.argmin(d))]
    out = lab == target
    return ndimage.binary_fill_holes(out)   # 火场内部的低烧毁“岛”也算在火场内


def offset_ring(perimeter: np.ndarray, res_m: float, ring_m: list[float]) -> np.ndarray:
    """火场外一圈未燃区，用来估计 dNBR 的季节/物候偏移。"""
    dist = ndimage.distance_transform_edt(~perimeter) * res_m
    return (dist >= ring_m[0]) & (dist <= ring_m[1])


def dnbr_offset(dnbr: np.ndarray, ring: np.ndarray) -> float:
    v = dnbr[ring & np.isfinite(dnbr)]
    return float(np.median(v)) if v.size else 0.0


def hotspot_perimeter(dnbr: np.ndarray, hot_rc: np.ndarray, res_m: float, threshold: float,
                      near_m: float = 750, far_m: float = 3000, min_patch: int = 20):
    """用 VIIRS 火点约束火烧范围：
    - 背景 = 离所有火点 > far_m 的像元 dNBR 中位数（扣掉干旱、物候造成的整体变化）；
    - 候选 = (dNBR - 背景) > 阈值，且离最近火点 ≤ near_m；
    - 只保留至少包含一个火点像元的连通区，再填补内部空洞。
    返回 (范围掩膜, 背景偏移, 离火点距离图)。"""
    shape = dnbr.shape
    hot = np.zeros(shape, bool)
    r, c = hot_rc[:, 0], hot_rc[:, 1]
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    hot[r[ok], c[ok]] = True
    if not hot.any():
        return np.zeros(shape, bool), 0.0, None
    dist = ndimage.distance_transform_edt(~hot) * res_m
    far = (dist > far_m) & np.isfinite(dnbr)
    bg = float(np.median(dnbr[far])) if far.sum() > 100 else 0.0
    cand = (np.nan_to_num(dnbr - bg, nan=-9) > threshold) & (dist <= near_m)
    cand = ndimage.binary_opening(cand, iterations=1)
    lab, n = ndimage.label(cand, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(shape, bool), bg, dist
    # 火点像元 375 m 有定位误差：在火点 1 个 VIIRS 像元范围内碰到的连通区都保留
    touch = np.unique(lab[(dist <= 375) & (lab > 0)])
    sizes = ndimage.sum(cand, lab, touch)
    keep = [t for t, s in zip(touch, sizes) if s >= min_patch]
    out = np.isin(lab, keep)
    out = ndimage.binary_closing(out, iterations=2)
    return ndimage.binary_fill_holes(out), bg, dist
