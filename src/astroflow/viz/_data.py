"""Data preparation shared by the plots. Plot code only reads SimulatedData / results."""
from __future__ import annotations

import json
from importlib import resources

import numpy as np
import pandas as pd

from ..constants import A_WGS84, B_WGS84
from ..dynamics import Propagator
from ..geodesy import ecef_to_geodetic, ray_ellipsoid, sees_point


def dense_track(data, n: int = 400, pad: float = 0.0):
    """Truth trajectory sampled densely over the observed arc (plus `pad` seconds each side-end)."""
    cfg = data.config
    t_end = float(data.frame_times[-1]) + pad
    times = np.linspace(0.0, max(t_end, 1.0), n)
    states, _ = Propagator(cfg.truth_force).propagate_many(data.truth_x_epoch, 0.0, times)
    return times, states


def to_ecef(earth, times, r_eci):
    return np.array([earth.rotation(t) @ r for t, r in zip(times, r_eci)])


def footprint_ecef(cfg, t: float, x: np.ndarray, n_edge: int = 24) -> np.ndarray:
    """Boundary of the camera's field of view on the ellipsoid, ECEF (M, 3); nan where rays miss."""
    cam = cfg.camera
    s = np.linspace(0.0, 1.0, n_edge, endpoint=False)
    w, h = cam.width, cam.height
    edge = np.vstack([np.column_stack([s * w, np.zeros_like(s)]),
                      np.column_stack([np.full_like(s, w), s * h]),
                      np.column_stack([w - s * w, np.full_like(s, h)]),
                      np.column_stack([np.zeros_like(s), h - s * h])])
    edge = np.vstack([edge, edge[:1]])
    R_att = cfg.attitude.rotation(t, x)
    R_e = cfg.earth.rotation(t)
    sat = R_e @ x[:3]
    # camera ray -> ECI (row form: @ R_att) -> ECEF (row form of R_e @ ray: @ R_e.T)
    d = (cam.unproject(edge) @ R_att) @ R_e.T
    pts, hit = ray_ellipsoid(sat, d)
    pts[~hit] = np.nan
    return pts


def ellipsoid_mesh(n_lat: int = 40, n_lon: int = 80):
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon = np.linspace(-np.pi, np.pi, n_lon)
    LON, LAT = np.meshgrid(lon, lat)
    return (A_WGS84 * np.cos(LAT) * np.cos(LON), A_WGS84 * np.cos(LAT) * np.sin(LON),
            B_WGS84 * np.sin(LAT))


def graticule_ecef(step_deg: int = 30, n: int = 60):
    """Meridians and parallels as a list of (n, 3) ECEF polylines."""
    from ..geodesy import geodetic_to_ecef
    lines = []
    u = np.linspace(-np.pi / 2, np.pi / 2, n)
    for lon in np.radians(np.arange(-180, 180, step_deg)):
        lines.append(geodetic_to_ecef(u, np.full_like(u, lon)))
    v = np.linspace(-np.pi, np.pi, 2 * n)
    for lat in np.radians(np.arange(-60, 61, step_deg)):
        lines.append(geodetic_to_ecef(np.full_like(v, lat), v))
    return lines


def world_topojson() -> dict:
    with resources.files("astroflow.viz").joinpath("data/world_110m.json").open() as f:
        return json.load(f)


def ground_track_df(data, n: int = 600) -> pd.DataFrame:
    """Sub-satellite lat/lon with a `segment` id that increments when the track wraps at +-180."""
    times, states = dense_track(data, n)
    lat, lon, _ = ecef_to_geodetic(to_ecef(data.config.earth, times, states[:, :3]))
    lon = np.degrees(lon); lat = np.degrees(lat)
    seg = np.concatenate([[0], np.cumsum(np.abs(np.diff(lon)) > 180)])
    return pd.DataFrame(dict(t=times, lat=lat, lon=lon, segment=seg))


def points_df(data) -> pd.DataFrame:
    """One row per observation, with truth labels for plotting."""
    obs = data.obs
    lat, lon, _ = ecef_to_geodetic(data.true_landmark_ecef) if len(obs) else (np.empty(0),) * 3
    return pd.DataFrame(dict(
        t=obs.t, frame=data.frame_index, u=obs.pixels[:, 0], v=obs.pixels[:, 1],
        sigma_px=obs.sigma_px, kind=np.where(data.is_outlier, "outlier", "inlier"),
        lat=np.degrees(lat), lon=np.degrees(lon)))
