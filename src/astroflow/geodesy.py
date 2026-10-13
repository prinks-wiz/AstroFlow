"""WGS84 geodesy: geodetic <-> ECEF, surface normals, horizon test, ray casting."""
from __future__ import annotations

import numpy as np

from .constants import A_WGS84, B_WGS84, E2_WGS84


def geodetic_to_ecef(lat, lon, h=0.0):
    """Radians and meters. Accepts scalars or arrays; returns (..., 3)."""
    lat, lon, h = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float), np.asarray(h, float))
    n = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - E2_WGS84) + h) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(r):
    """Return (lat, lon, h) in radians/meters for ECEF points (..., 3). Iterative, ~1e-12 rad."""
    r = np.asarray(r, float)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - E2_WGS84))
    for _ in range(10):
        n = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * np.sin(lat) ** 2)
        # height form that is well-conditioned at the poles too
        h = p * np.cos(lat) + z * np.sin(lat) - A_WGS84**2 / n
        lat = np.arctan2(z, p * (1.0 - E2_WGS84 * n / (n + h)))
    n = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * np.sin(lat) ** 2)
    h = p * np.cos(lat) + z * np.sin(lat) - A_WGS84**2 / n
    return lat, lon, h


def surface_normal(lat, lon):
    """Unit geodetic 'up' vector in ECEF, (..., 3)."""
    lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def sees_point(sat_ecef, points_ecef, margin: float = 0.0):
    """True where a ground point can see the satellite above its local horizon.

    The ellipsoid is convex, so the straight line from a surface point to the satellite
    is unobstructed iff it leaves through the outward half-space at that point.
    `margin` (m) requires the satellite to be that far above the local tangent plane.
    """
    pts = np.atleast_2d(np.asarray(points_ecef, float))
    lat, lon, _ = ecef_to_geodetic(pts)
    up = surface_normal(lat, lon)
    return np.einsum("ij,ij->i", up, np.asarray(sat_ecef, float) - pts) > margin


def ray_ellipsoid(origin, directions):
    """Intersect rays with the WGS84 ellipsoid surface (near hit).

    origin: (3,) ECEF; directions: (N, 3) ECEF (need not be unit).
    Returns points (N, 3) and a boolean mask of rays that hit.
    """
    o = np.asarray(origin, float)
    d = np.atleast_2d(np.asarray(directions, float))
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    s = np.array([1.0 / A_WGS84, 1.0 / A_WGS84, 1.0 / B_WGS84])
    os_, ds = o * s, d * s
    a = np.einsum("ij,ij->i", ds, ds)
    b = 2.0 * ds @ os_
    c = os_ @ os_ - 1.0
    disc = b * b - 4 * a * c
    hit = disc >= 0
    t = (-b - np.sqrt(np.where(hit, disc, 0.0))) / (2 * a)
    hit &= t > 0
    return o + t[:, None] * d, hit
