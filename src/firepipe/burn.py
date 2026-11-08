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
