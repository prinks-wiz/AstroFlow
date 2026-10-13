"""3D orbit views without WebGL.

Altair has no 3D marks, so the orthographic projection is done in the chart (Vega expressions)
and re-evaluated as the azimuth / elevation sliders move. Everything is drawn as ordinary 2D
vector graphics, so it works in browsers and environments without WebGL.

Earth is a disc (an orthographic view of a sphere is always a circle); anything that lies
behind it from the current viewpoint is hidden.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from ..constants import R_EARTH
from ..dynamics import Propagator
from ._data import dense_track, footprint_ecef, graticule_ecef

KM = 1e-3
SIZE_PX = 640
_BG = "#0b1620"

# screen axes for azimuth az, elevation el (degrees): u right, v up, d toward the viewer
_RAD = "PI/180"
_U = f"-sin(az*{_RAD})*datum.x + cos(az*{_RAD})*datum.y"
_V = (f"-sin(el*{_RAD})*cos(az*{_RAD})*datum.x - sin(el*{_RAD})*sin(az*{_RAD})*datum.y"
      f" + cos(el*{_RAD})*datum.z")
_D = (f"cos(el*{_RAD})*cos(az*{_RAD})*datum.x + cos(el*{_RAD})*sin(az*{_RAD})*datum.y"
      f" + sin(el*{_RAD})*datum.z")


def _rot_angle(earth, t: float) -> float:
    R = earth.rotation(t)
    return float(np.arctan2(R[0, 1], R[0, 0]))  # ECI->ECEF rotation about z


def _default_view(data, el: float = 20.0, off_normal_deg: float = 55.0):
    """Azimuth/elevation that looks at the orbit plane obliquely (not edge-on, not face-on),
    so the orbit's inclination and the near/far sides are visible."""
    n = np.cross(data.truth_x_epoch[:3], data.truth_x_epoch[3:])
    n /= np.linalg.norm(n)
    best = None
    for az in np.arange(-180, 180, 5.0):
        a, e = np.radians(az), np.radians(el)
        c = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        err = abs(np.degrees(np.arccos(np.clip(abs(c @ n), 0, 1))) - off_normal_deg)
        if best is None or err < best[0]:
            best = (err, az)
    return best[1], el


def _project(chart: alt.Chart, rotate_grid: str | None = None, hide_behind_earth: bool = True) -> alt.Chart:
    """Add the projection transforms (and optionally rotate points about z first)."""
    c = chart
    if rotate_grid:
        c = (c.transform_calculate(x=f"datum.gx*cos({rotate_grid}) - datum.gy*sin({rotate_grid})",
                                   y=f"datum.gx*sin({rotate_grid}) + datum.gy*cos({rotate_grid})",
                                   z="datum.gz"))
    c = c.transform_calculate(u=_U, v=_V, d=_D)
    if hide_behind_earth:
        c = c.transform_filter(f"!(datum.d < 0 && datum.u*datum.u + datum.v*datum.v < {(R_EARTH * KM) ** 2})")
    return c


def _build(data, animated: bool, n_dense: int, n_time: int) -> alt.LayerChart:
    cfg = data.config
    L = 1.25 * max(np.linalg.norm(data.truth_states[:, :3], axis=1).max(), R_EARTH) * KM
    xy = dict(
        x=alt.X("u:Q", scale=alt.Scale(domain=[-L, L]), axis=None),
        y=alt.Y("v:Q", scale=alt.Scale(domain=[-L, L]), axis=None))

    # --- view and time controls
    az0, el0 = _default_view(data)
    az = alt.param(name="az", value=az0, bind=alt.binding_range(min=-180, max=180, step=1, name="azimuth (deg) "))
    el = alt.param(name="el", value=el0, bind=alt.binding_range(min=-90, max=90, step=1, name="elevation (deg) "))
    params = [az, el]

    # --- Earth disc: radius in px = R * (SIZE_PX / 2L); a circle symbol's size is the area of its bounding square, so r = sqrt(size)/2
    r_px = R_EARTH * KM * SIZE_PX / (2 * L)
    earth = alt.Chart(pd.DataFrame(dict(u=[0.0], v=[0.0]))).mark_circle(
        size=float((2 * r_px) ** 2), color="#1d4f73", opacity=1.0).encode(**xy)

    # --- lat/lon grid, rotating with the Earth when animated
    th0 = _rot_angle(cfg.earth, 0.0)
    t_end = float(max(data.frame_times[-1], 1.0))
    dt = t_end / (n_time - 1)
    rate = (_rot_angle(cfg.earth, dt) - th0) / dt
    # ECI->ECEF angle th means ECEF->ECI rotates by -th; use the inverse sign here
    grid_rows = []
    for i, line in enumerate(graticule_ecef(30, 90)):
        grid_rows.append(pd.DataFrame(dict(gx=line[:, 0] * KM, gy=line[:, 1] * KM, gz=line[:, 2] * KM, line=i)))
    grid = pd.concat(grid_rows, ignore_index=True)
    ang = f"(-({th0} + {rate * dt} * tk))" if animated else f"(-{th0})"
    grid_chart = _project(alt.Chart(grid).mark_circle(size=6, color="#8fb8d4", opacity=0.6), ang).encode(**xy)

    # --- orbit path and frame markers
    times, states = dense_track(data, n_dense)
    orbit = pd.DataFrame(dict(x=states[:, 0] * KM, y=states[:, 1] * KM, z=states[:, 2] * KM))
    orbit_chart = _project(alt.Chart(orbit).mark_circle(size=9, color="#e8a33d"), None).encode(**xy)
    fs = data.truth_states
    frames = pd.DataFrame(dict(x=fs[:, 0] * KM, y=fs[:, 1] * KM, z=fs[:, 2] * KM, frame=np.arange(len(fs)),
                               t=data.frame_times))
    frame_chart = _project(alt.Chart(frames).mark_circle(size=40, stroke="#0b1620", strokeWidth=0.8), None).encode(
        color=alt.Color("t:Q", scale=alt.Scale(scheme="viridis"), title="frame time (s)"),
        tooltip=["frame", alt.Tooltip("t:Q", format=".0f")], **xy)
    layers = [earth, grid_chart, orbit_chart, frame_chart]

    # --- satellite and camera footprint at the slider time
    if animated:
        tk = alt.param(name="tk", value=0, bind=alt.binding_range(min=0, max=n_time - 1, step=1, name="time step "))
        params.append(tk)
        ts = np.linspace(0.0, t_end, n_time)
        sx, _ = Propagator(cfg.truth_force).propagate_many(data.truth_x_epoch, 0.0, ts)
        sat = pd.DataFrame(dict(x=sx[:, 0] * KM, y=sx[:, 1] * KM, z=sx[:, 2] * KM, k=np.arange(n_time), t=ts))
        sat_chart = _project(alt.Chart(sat).mark_circle(size=130, color="#ffffff", stroke="#e8a33d", strokeWidth=2)
                             .transform_filter("datum.k == tk"), None).encode(
            tooltip=[alt.Tooltip("t:Q", format=".0f", title="t (s)")], **xy)
        fp_rows = []
        for k, (t, x) in enumerate(zip(ts, sx)):
            f = footprint_ecef(cfg, t, x) @ cfg.earth.rotation(t)  # ECEF -> ECI
            fp_rows.append(pd.DataFrame(dict(x=f[:, 0] * KM, y=f[:, 1] * KM, z=f[:, 2] * KM, k=k,
                                             order=np.arange(len(f)))))
        fp = pd.concat(fp_rows, ignore_index=True).dropna()
        fp_chart = _project(alt.Chart(fp).mark_line(color="#ff6b6b", strokeWidth=2.5)
                            .transform_filter("datum.k == tk"), None).encode(order="order:Q", **xy)
        layers += [fp_chart, sat_chart]

    title = ("Orbit animation: drag the sliders to rotate the view and move the satellite"
             if animated else "True orbit: drag the sliders to rotate the view")
    chart = (alt.layer(*layers).add_params(*params)
             .properties(width=SIZE_PX, height=SIZE_PX, title=alt.TitleParams(title, color="#dfe8ee"))
             .configure_view(fill=_BG, strokeWidth=0)
             .configure_legend(labelColor="#dfe8ee", titleColor="#dfe8ee"))
    return chart


def plot_orbit_3d(data, n_dense: int = 500) -> alt.LayerChart:
    """Earth, the true orbit over the observed arc and the satellite at each frame time.
    Sliders rotate the view (azimuth, elevation). Works without WebGL."""
    return _build(data, animated=False, n_dense=n_dense, n_time=2)


def animate_orbit(data, n_time: int = 120, n_dense: int = 500) -> alt.LayerChart:
    """As `plot_orbit_3d`, plus a time slider: the satellite moves along the orbit, the lat/lon
    grid turns with the Earth, and the camera footprint is shown on the ground."""
    return _build(data, animated=True, n_dense=n_dense, n_time=n_time)
