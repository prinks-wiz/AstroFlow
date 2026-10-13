"""Interactive 3D orbit views (Plotly): static plot and an animation with the Earth rotating."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ._data import dense_track, ellipsoid_mesh, footprint_ecef, graticule_ecef

_EARTH_COLORS = [[0, "#1d3b53"], [1, "#2d5a7b"]]


def _scene(fig, title):
    fig.update_layout(
        title=title, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h"),
        scene=dict(aspectmode="data", xaxis_title="x ECI (m)", yaxis_title="y ECI (m)", zaxis_title="z ECI (m)"))


def _earth_trace():
    x, y, z = ellipsoid_mesh()
    return go.Surface(x=x, y=y, z=z, colorscale=_EARTH_COLORS, showscale=False, opacity=1.0,
                      hoverinfo="skip", name="Earth", lighting=dict(ambient=0.8, diffuse=0.4))


def _graticule_xyz(earth, t):
    """Graticule rotated into ECI at time t, as one NaN-separated polyline."""
    R_ie = earth.rotation(t).T
    xs, ys, zs = [], [], []
    for line in graticule_ecef():
        p = line @ R_ie.T
        xs += list(p[:, 0]) + [np.nan]; ys += list(p[:, 1]) + [np.nan]; zs += list(p[:, 2]) + [np.nan]
    return xs, ys, zs


def plot_orbit_3d(data, n_dense: int = 400) -> go.Figure:
    """Earth, the true orbit over the observed arc, and the satellite at each frame time."""
    cfg = data.config
    times, states = dense_track(data, n_dense)
    fig = go.Figure(_earth_trace())
    gx, gy, gz = _graticule_xyz(cfg.earth, 0.0)
    fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", line=dict(color="#7aa6c4", width=1),
                               hoverinfo="skip", name="Lat/lon grid (t=0)"))
    fig.add_trace(go.Scatter3d(x=states[:, 0], y=states[:, 1], z=states[:, 2], mode="lines",
                               line=dict(color="#e8a33d", width=4), name="Orbit"))
    fs = data.truth_states
    fig.add_trace(go.Scatter3d(
        x=fs[:, 0], y=fs[:, 1], z=fs[:, 2], mode="markers", name="Frames",
        marker=dict(size=4, color=data.frame_times, colorscale="Viridis",
                    colorbar=dict(title="t (s)", len=0.5)),
        text=[f"frame {k}, t={t:.0f} s" for k, t in enumerate(data.frame_times)], hoverinfo="text"))
    _scene(fig, "True orbit (ECI)")
    return fig


def animate_orbit(data, n_frames: int = 90) -> go.Figure:
    """Satellite moving along the orbit, the lat/lon grid rotating with the Earth, and the
    camera footprint on the ground."""
    cfg = data.config
    times, states = dense_track(data, 400)
    anim_t = np.linspace(0.0, times[-1], n_frames)
    anim_x, _ = __import__("astroflow.dynamics", fromlist=["Propagator"]).Propagator(cfg.truth_force).propagate_many(
        data.truth_x_epoch, 0.0, anim_t)

    def foot_eci(t, x):
        f = footprint_ecef(cfg, t, x)
        return f @ cfg.earth.rotation(t)  # ECEF -> ECI (row form of R^T f)

    gx, gy, gz = _graticule_xyz(cfg.earth, 0.0)
    f0 = foot_eci(anim_t[0], anim_x[0])
    fig = go.Figure([
        _earth_trace(),
        go.Scatter3d(x=states[:, 0], y=states[:, 1], z=states[:, 2], mode="lines",
                     line=dict(color="#e8a33d", width=3), name="Orbit"),
        go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", line=dict(color="#7aa6c4", width=1), hoverinfo="skip", name="Grid"),
        go.Scatter3d(x=[anim_x[0, 0]], y=[anim_x[0, 1]], z=[anim_x[0, 2]], mode="markers",
                     marker=dict(size=6, color="#ffffff"), name="Satellite"),
        go.Scatter3d(x=f0[:, 0], y=f0[:, 1], z=f0[:, 2], mode="lines", line=dict(color="#ff6b6b", width=4),
                     name="Camera footprint"),
    ])
    frames = []
    for t, x in zip(anim_t, anim_x):
        gx, gy, gz = _graticule_xyz(cfg.earth, t)
        f = foot_eci(t, x)
        frames.append(go.Frame(name=f"{t:.0f}", traces=[2, 3, 4], data=[
            go.Scatter3d(x=gx, y=gy, z=gz),
            go.Scatter3d(x=[x[0]], y=[x[1]], z=[x[2]]),
            go.Scatter3d(x=f[:, 0], y=f[:, 1], z=f[:, 2])]))
    fig.frames = frames
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=0.02, buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True)]),
            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")])])],
        sliders=[dict(active=0, x=0.2, len=0.75, currentvalue=dict(prefix="t = ", suffix=" s"),
                      steps=[dict(label=f.name, method="animate",
                                  args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for f in frames])])
    _scene(fig, "Orbit animation (ECI; grid rotates with the Earth)")
    return fig
