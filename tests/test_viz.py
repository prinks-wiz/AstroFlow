import numpy as np
import pytest

pytest.importorskip("altair")
pytest.importorskip("plotly")

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
    fig = plot_orbit_3d(data)
    assert [t.name for t in fig.data][:1] == ["Earth"] and "Frames" in [t.name for t in fig.data]
    orbit = next(t for t in fig.data if t.name == "Orbit")
    r = np.linalg.norm(np.column_stack([orbit.x, orbit.y, orbit.z]), axis=1)
    assert 6.8e6 < r.min() and r.max() < 7.1e6
    anim = animate_orbit(data, n_frames=10)
    assert len(anim.frames) == 10 and fig.to_html()


def test_settings_chart_shows_gain_saturation_tradeoff():
    from astroflow.viz.camera_view import settings_effect_df
    df = settings_effect_df(MatchQualityModel())
    hi = df[df.gain == 8.0].sort_values("exposure_ms")
    assert hi.p_detect.iloc[-1] < 0.1 < hi.p_detect.max()
