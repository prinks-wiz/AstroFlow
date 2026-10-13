"""Outlier generators: gross wrong matches."""
from __future__ import annotations

import numpy as np

from ..camera import Camera
from ..geodesy import ecef_to_geodetic, geodetic_to_ecef


def random_pixel(rng, n: int, camera: Camera):
    """Observed pixel uniform over the image, unrelated to the reported landmark."""
    return np.column_stack([rng.uniform(0, camera.width, n), rng.uniform(0, camera.height, n)])


def displaced_landmark(rng, true_ecef, offset_range_m):
    """Valid pixel paired with a landmark that is really somewhere else: the true point
    moved horizontally by a log-uniform distance in a random direction (a near-miss mismatch)."""
    true_ecef = np.atleast_2d(true_ecef)
    lat, lon, h = ecef_to_geodetic(true_ecef)
    d = np.exp(rng.uniform(np.log(offset_range_m[0]), np.log(offset_range_m[1]), len(true_ecef)))
    az = rng.uniform(0, 2 * np.pi, len(true_ecef))
    # spherical-earth displacement is fine for generating a mismatch
    R = 6371e3
    dlat = d * np.cos(az) / R
    dlon = d * np.sin(az) / (R * np.maximum(np.cos(lat), 1e-3))
    return geodetic_to_ecef(np.clip(lat + dlat, -np.pi / 2, np.pi / 2), lon + dlon, h)
