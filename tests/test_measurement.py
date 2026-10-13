import numpy as np
import pytest

from astroflow.attitude import NadirAttitude, TelemetryAttitude
from astroflow.camera import PinholeCamera
from astroflow.constants import R_EARTH
from astroflow.dynamics import J2Gravity, Propagator, kepler_to_cartesian
from astroflow.errors import Observations, landmark_sigma_to_pixels, total_pixel_sigma
from astroflow.frames import SimpleEarthRotation
from astroflow.geodesy import ray_ellipsoid
from astroflow.measurement import predict_pixels

CAM = PinholeCamera.from_fov(k1=-0.02)
EARTH = SimpleEarthRotation(theta0=0.4)
PROP = Propagator(J2Gravity())
X0 = kepler_to_cartesian(R_EARTH + 550e3, 0.002, np.radians(60), 0.3, 1.0, 0.5)


def _scene(att, times, rng, per=6):
    """Landmarks obtained by casting rays from pixel samples, so truth pixels are known."""
    states, _ = PROP.propagate_many(X0, 0.0, times)
    lm, uv_true, tt = [], [], []
    for tk, xk in zip(times, states):
        R = att.rotation(tk, xk)
        uv = np.column_stack([rng.uniform(100, CAM.width - 100, per), rng.uniform(100, CAM.height - 100, per)])
        d_ecef = (CAM.unproject(uv) @ R) @ EARTH.rotation(tk).T
        sat_ecef = EARTH.rotation(tk) @ xk[:3]
        pts, hit = ray_ellipsoid(sat_ecef, d_ecef)
        assert hit.all()
        lm.append(pts); uv_true.append(uv); tt += [tk] * per
    return np.array(tt), np.vstack(lm), np.vstack(uv_true)


def test_noise_free_pixels_reproject_exactly():
    rng = np.random.default_rng(0)
    t, lm, uv_true = _scene(NadirAttitude(yaw_deg=12), np.linspace(0, 1500, 6), rng)
    pred = predict_pixels(X0, 0.0, t, lm, PROP, EARTH, NadirAttitude(yaw_deg=12), CAM, jacobian=False)
    assert pred.valid.all()
    assert np.abs(pred.pixels - uv_true).max() < 1e-9  # measured ~2e-12 px


@pytest.mark.parametrize("att", [NadirAttitude(yaw_deg=5),
                                 TelemetryAttitude([0, 2000], [[0, 0, 0, 1], [0, 0, 0.0, 1]])])
def test_jacobian_matches_finite_difference(att):
    rng = np.random.default_rng(3)
    times = np.linspace(0, 1500, 5)
    if isinstance(att, TelemetryAttitude):
        times = np.linspace(0, 100, 5)  # fixed attitude: keep the Earth in view  # nadir-ish constant attitude from the initial state
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(NadirAttitude().rotation(0.0, X0)).as_quat()
        att = TelemetryAttitude([0, 2000], [q, q])
    t, lm, _ = _scene(att, times, rng)
    pred = predict_pixels(X0, 0.0, t, lm, PROP, EARTH, att, CAM)
    for k in range(6):
        h = 1.0 if k < 3 else 1e-3
        d = np.zeros(6); d[k] = h
        up = predict_pixels(X0 + d, 0.0, t, lm, PROP, EARTH, att, CAM, jacobian=False).pixels
        dn = predict_pixels(X0 - d, 0.0, t, lm, PROP, EARTH, att, CAM, jacobian=False).pixels
        fd = (up - dn) / (2 * h)
        scale = np.abs(pred.jacobian[:, :, k]).max()
        assert np.abs(pred.jacobian[:, :, k] - fd).max() < 1e-4 * scale, k


def test_behind_camera_is_invalid():
    # a point above the satellite is behind a nadir-pointing camera.
    # (Occlusion by the Earth is NOT part of `valid`; the simulator applies geodesy.sees_point.)
    up = X0[:3] / np.linalg.norm(X0[:3]) * (R_EARTH + 3000e3)
    pred = predict_pixels(X0, 0.0, [0.0], (EARTH.rotation(0.0) @ up)[None], PROP, EARTH, NadirAttitude(), CAM)
    assert not pred.valid[0] and np.isnan(pred.pixels).all()


def test_sigma_conversion():
    # 10 m at 500 km with f = 1662 px -> 0.033 px
    f = CAM.focal_px
    assert landmark_sigma_to_pixels(10.0, 500e3, f) == pytest.approx(f * 2e-5)
    assert total_pixel_sigma(0.5, 0.0, 500e3, f) == pytest.approx(0.5)
    assert total_pixel_sigma(0.3, 100.0, 500e3, f) == pytest.approx(np.hypot(0.3, f * 2e-4))


def test_observations_validate_and_subset():
    o = Observations([0, 1, 2], np.zeros((3, 2)), np.zeros((3, 3)), np.ones(3))
    assert len(o.subset(np.array([True, False, True]))) == 2
