import numpy as np
import pytest

from astroflow.constants import R_EARTH
from astroflow.dynamics import J2Gravity, Propagator
from astroflow.errors import landmark_sigma_to_pixels
from astroflow.geodesy import sees_point
from astroflow.measurement import predict_pixels
from astroflow.sensor import CameraSettings, MatchQualityModel
from astroflow.sim import SimConfig, Simulator


def gen(**kw):
    return Simulator(SimConfig(**kw)).generate()


def residuals(d, cfg):
    """observed - predicted using the true state and the *reported* landmarks."""
    pred = predict_pixels(d.truth_x_epoch, 0.0, d.obs.t, d.obs.landmark_ecef, Propagator(cfg.truth_force),
                          cfg.earth, cfg.attitude, cfg.camera, jacobian=False)
    return d.obs.pixels - pred.pixels, pred


def test_seeded_reproducibility_bitwise():
    a, b, c = gen(seed=5, outlier_fraction=0.2), gen(seed=5, outlier_fraction=0.2), gen(seed=6, outlier_fraction=0.2)
    for f in ("t", "pixels", "landmark_ecef", "sigma_px"):
        assert np.array_equal(getattr(a.obs, f), getattr(b.obs, f))
    assert np.array_equal(a.is_outlier, b.is_outlier)
    assert not np.array_equal(a.obs.pixels, c.obs.pixels)


@pytest.mark.parametrize("f", [0.0, 0.1, 0.3, 0.5])
def test_outlier_fraction_hits_target(f):
    d = gen(outlier_fraction=f, inliers_per_frame=40, n_frames=10)
    assert d.is_outlier.mean() == pytest.approx(f, abs=0.02)
    assert (~d.is_outlier).sum() == pytest.approx(40 * 10, rel=0.02)  # every in-view candidate is detected


def test_inliers_consistent_with_configured_sigma():
    cfg = SimConfig(inliers_per_frame=120, n_frames=15, pixel_sigma=0.6, landmark_sigma_m=40.0, seed=1)
    d = Simulator(cfg).generate()
    res, pred = residuals(d, cfg)
    z = (res / d.obs.sigma_px[:, None])[~d.is_outlier]
    assert z.std() == pytest.approx(1.0, abs=0.06)           # sigma_px reported matches reality
    assert abs(z.mean()) < 0.05
    # and the reported sigma is the combination of image noise and landmark error
    expect = np.hypot(0.6, landmark_sigma_to_pixels(40.0, pred.range_m, cfg.camera.focal_px))
    assert np.allclose(d.obs.sigma_px[~d.is_outlier], expect, rtol=0.02)


def test_noise_free_inliers_reproject_exactly():
    cfg = SimConfig(pixel_sigma=0.0, landmark_sigma_m=0.0)
    d = Simulator(cfg).generate()
    res, _ = residuals(d, cfg)
    assert np.abs(res).max() < 1e-8


def test_all_landmarks_visible():
    cfg = SimConfig(outlier_fraction=0.3)
    d = Simulator(cfg).generate()
    for k, tk in enumerate(d.frame_times):
        sat = cfg.earth.rotation(tk) @ d.truth_states[k, :3]
        m = d.frame_index == k
        assert sees_point(sat, d.true_landmark_ecef[m]).all()
    ins = ~d.is_outlier
    assert cfg.camera.in_image(d.true_pixels[ins], 0).all()


def test_outlier_kinds_are_real_mismatches():
    cfg = SimConfig(outlier_fraction=0.4, pixel_sigma=0.3, landmark_sigma_m=5.0)
    d = Simulator(cfg).generate()
    res, _ = residuals(d, cfg)
    err = np.linalg.norm(res, axis=1)
    assert np.median(err[~d.is_outlier]) < 3
    assert np.median(err[d.is_outlier]) > 20          # clearly wrong versus 0.3 px noise
    assert np.all(d.obs.sigma_px[d.is_outlier] > 0)


def test_outlier_sigma_inflate_reports_lower_confidence():
    d = gen(outlier_fraction=0.3, outlier_sigma_inflate=5.0)
    assert d.obs.sigma_px[d.is_outlier].mean() > 4 * d.obs.sigma_px[~d.is_outlier].mean()


def test_camera_settings_change_the_output():
    q = MatchQualityModel()
    short = gen(quality=q, settings=CameraSettings(exposure_s=0.003), inliers_per_frame=60)
    long_ = gen(quality=q, settings=CameraSettings(exposure_s=0.06), inliers_per_frame=60)
    assert long_.obs.sigma_px.mean() > short.obs.sigma_px.mean()    # blur raises sigma
    off = gen(quality=q, settings=CameraSettings(wb_gains=(1.8, 1.0, 0.5)), outlier_fraction=0.0)
    assert off.is_outlier.mean() > 0.05                              # white-balance mismatch adds outliers
    assert gen(quality=q, settings=CameraSettings(wb_gains=(1, 1, 1)), outlier_fraction=0.0).is_outlier.mean() == 0
    # per-frame schedule is honored
    sched = gen(quality=q, settings=[CameraSettings(exposure_s=0.003), CameraSettings(exposure_s=0.06)], n_frames=2,
                inliers_per_frame=60)
    s0, s1 = [sched.obs.sigma_px[sched.frame_index == k].mean() for k in (0, 1)]
    assert s1 > s0


def test_night_side_yields_no_points():
    # sun on the opposite side of the Earth from the nadir point at t=0
    from astroflow.illumination import SimpleSun
    from astroflow.dynamics import kepler_to_cartesian
    cfg = SimConfig(quality=MatchQualityModel(), n_frames=1)
    x0 = kepler_to_cartesian(*cfg.orbit)
    cfg.sun = SimpleSun.towards(x0[:3], offset=np.pi)  # anti-sunward over the sub-satellite point
    d = Simulator(cfg).generate()
    assert len(d.obs) < 0.25 * cfg.inliers_per_frame
