"""Ground track and camera footprints on a world map (Altair)."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from ._data import footprint_ecef, ground_track_df, points_df, world_topojson
from ..geodesy import ecef_to_geodetic

KIND_COLORS = alt.Scale(domain=["inlier", "outlier"], range=["#2a9d8f", "#d1495b"])


def _footprint_df(data, max_frames: int = 40) -> pd.DataFrame:
    cfg = data.config
    idx = np.unique(np.linspace(0, len(data.frame_times) - 1, min(max_frames, len(data.frame_times))).astype(int))
    rows = []
    for k in idx:
        f = footprint_ecef(cfg, data.frame_times[k], data.truth_states[k])
        lat, lon, _ = ecef_to_geodetic(f)
        lon = np.degrees(lon); lat = np.degrees(lat)
        seg = np.concatenate([[0], np.cumsum(np.abs(np.diff(lon)) > 180)])
        rows.append(pd.DataFrame(dict(frame=k, order=np.arange(len(lat)), lat=lat, lon=lon,
                                      detail=[f"{k}-{s}" for s in seg])))
    return pd.concat(rows, ignore_index=True)


def plot_ground_track(data, width: int = 900, show_footprints: bool = True) -> alt.Chart:
    """World map with the sub-satellite track (orange), camera footprints (blue) and matched
    landmarks: inliers teal, outliers red. Points are drawn at the true ground location of the
    scene point behind each match."""
    topo = alt.Data(values=world_topojson(), format=alt.DataFormat(type="topojson", feature="countries"))
    height = int(width / 2)
    proj = dict(type="equirectangular")
    base = alt.Chart(topo).mark_geoshape(fill="#e9eef2", stroke="#b7c3cc", strokeWidth=0.4)
    track = ground_track_df(data)
    line = alt.Chart(track).mark_line(color="#e8a33d", strokeWidth=2.5).encode(
        longitude="lon:Q", latitude="lat:Q", detail="segment:N", order="t:Q")
    layers = [base, line]
    if show_footprints:
        fp = _footprint_df(data)
        layers.append(alt.Chart(fp).mark_line(color="#4361ee", opacity=0.35, strokeWidth=1).encode(
            longitude="lon:Q", latitude="lat:Q", detail="detail:N", order="order:Q"))
    pts = points_df(data)
    if len(pts):
        layers.append(alt.Chart(pts).mark_circle(size=22, opacity=0.8).encode(
            longitude="lon:Q", latitude="lat:Q", color=alt.Color("kind:N", scale=KIND_COLORS, title="match"),
            tooltip=["frame", "kind", alt.Tooltip("sigma_px:Q", format=".2f")]))
    return alt.layer(*layers).properties(width=width, height=height, title="Ground track, footprints and matched landmarks").project(**proj)
