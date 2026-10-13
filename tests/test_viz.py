import numpy as np
import pytest

pytest.importorskip("altair")

from astroflow.constants import A_WGS84, B_WGS84
from astroflow.sensor import MatchQualityModel
from astroflow.sim import SimConfig, Simulator
from astroflow.viz import (animate_orbit, plot_ground_track, plot_orbit_3d, plot_pixel_tracks,
                           plot_settings_effect)
from astroflow.viz._data import (dense_track, footprint_ecef, ground_track_df, points_df, world_topojson)


@pytest.fixture(scope="module")
def data():
    return Simulator(SimConfig(n_frames=8, frame_dt=300.0, outlier_fraction=0.25, quality=MatchQualityModel())).generate()


def test_dense_track_matches_frame_states(data):
    times, states = dense_track(data, n=(len(data.frame_times) - 1) * 10 + 1)  # includes every frame time
    idx = np.searchsorted(times, data.frame_times)
    assert np.allclose(times[idx], data.frame_times, atol=1e-6)
    assert np.allclose(states[idx], data.truth_states, rtol=1e-7, atol=1e-3)


def test_footprint_lies_on_ellipsoid_and_surrounds_nadir(data):
    f = footprint_ecef(data.config, data.frame_times[2], data.truth_states[2])
    assert not np.isnan(f).any()
    q = (f[:, 0] ** 2 + f[:, 1] ** 2) / A_WGS84**2 + f[:, 2] ** 2 / B_WGS84**2
    assert np.allclose(q, 1.0, atol=1e-9)


def test_ground_track_df_segments_at_dateline(data):
    df = ground_track_df(data, n=300)
    assert df.lat.abs().max() < 65 and df.lon.abs().max() <= 180
    assert (df.groupby("segment").lon.apply(lambda s: s.diff().abs().max()).dropna() < 180).all()


def test_points_df_labels(data):
    df = points_df(data)
    assert len(df) == len(data.obs) and set(df.kind) <= {"inlier", "outlier"}
    assert (df.kind == "outlier").sum() == data.is_outlier.sum()


def test_world_map_is_bundled():
    assert "countries" in world_topojson()["objects"]


def test_plots_build_and_serialize(data):
    for chart in (plot_ground_track(data), plot_pixel_tracks(data), plot_settings_effect()):
        assert chart.to_dict()  # valid Vega-Lite spec


def _spec(chart):
    return chart.to_dict()


def test_3d_views_need_no_webgl_and_have_the_right_controls(data):
    static, anim = _spec(plot_orbit_3d(data)), _spec(animate_orbit(data, n_time=10))
    names = lambda sp: {p["name"] for p in sp["params"]} if "params" in sp else {
        p["name"] for lay in sp["layer"] for p in lay.get("params", [])}
    assert names(static) >= {"az", "el"} and "tk" not in names(static)
    assert names(anim) >= {"az", "el", "tk"}
    import json
    assert "scatter3d" not in json.dumps(anim).lower()  # plain 2D marks only


def test_default_view_is_oblique_to_the_orbit_plane(data):
    from astroflow.viz.orbit3d import _default_view
    az, el = _default_view(data)
    c = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)), np.cos(np.radians(el)) * np.sin(np.radians(az)),
                  np.sin(np.radians(el))])
    n = np.cross(data.truth_x_epoch[:3], data.truth_x_epoch[3:]); n /= np.linalg.norm(n)
    assert 45 < np.degrees(np.arccos(abs(c @ n))) < 65


def test_projection_expressions_match_numpy_projection(data):
    """The Vega expressions encode u = p.e1, v = p.e2, d = p.c; check the orthonormal frame they imply."""
    az, el = np.radians(37.0), np.radians(21.0)
    c = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    e1 = np.array([-np.sin(az), np.cos(az), 0.0])
    e2 = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
    assert np.allclose([e1 @ e1, e2 @ e2, c @ c, e1 @ e2, e1 @ c, e2 @ c], [1, 1, 1, 0, 0, 0], atol=1e-12)
    assert np.allclose(np.cross(e1, e2), c)  # right-handed: u right, v up, d toward the viewer


def test_settings_chart_shows_gain_saturation_tradeoff():
    from astroflow.viz.camera_view import settings_effect_df
    df = settings_effect_df(MatchQualityModel())
    hi = df[df.gain == 8.0].sort_values("exposure_ms")
    assert hi.p_detect.iloc[-1] < 0.1 < hi.p_detect.max()
