"""统一网格：每个火场一个 UTM 30 m 网格，所有图层都对齐到它。"""
from __future__ import annotations

import math
from dataclasses import dataclass

from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180) / 6)) + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class FireGrid:
    fire_id: str
    epsg: int
    bbox_lonlat: tuple[float, float, float, float]
    geobox: GeoBox

    @classmethod
    def from_fire(cls, fire: dict, resolution_m: float, half_deg: float,
                  boundary_bounds: tuple | None = None, buffer_m: float = 1000):
        lon, lat = float(fire["center_lon"]), float(fire["center_lat"])
        if boundary_bounds:
            minx, miny, maxx, maxy = boundary_bounds
            pad = buffer_m / 111_000
            bbox = (minx - pad, miny - pad, maxx + pad, maxy + pad)
        else:
            bbox = (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)
        epsg = utm_epsg(lon, lat)
        gb = GeoBox.from_bbox(BoundingBox(*bbox, crs="EPSG:4326").to_crs(f"EPSG:{epsg}"),
                              crs=f"EPSG:{epsg}", resolution=resolution_m)
        return cls(fire["fire_id"], epsg, bbox, gb)
