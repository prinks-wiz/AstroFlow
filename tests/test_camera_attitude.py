import numpy as np
import pytest

from astroflow.attitude import NadirAttitude, TelemetryAttitude
from astroflow.camera import PinholeCamera
from astroflow.dynamics import kepler_to_cartesian
from astroflow.constants import R_EARTH
from scipy.spatial.transform import Rotation

CAMS = [PinholeCamera.from_fov(), PinholeCamera.from_fov(k1=-0.05, k2=0.01)]


@pytest.mark.parametrize("cam", CAMS)
def test_project_unproject_roundtrip(cam):
    rng = np.random.default_rng(0)
    uv = np.column_stack([rng.uniform(0, cam.width, 200), rng.uniform(0, cam.height, 200)])
    rays = cam.unproject(uv)
    assert np.allclose(np.linalg.norm(rays, axis=1), 1)
    assert np.allclose(cam.project(rays * 3.7), uv, atol=1e-8)


@pytest.mark.parametrize("cam", CAMS)
def test_projection_jacobian_matches_finite_difference(cam):
    rng = np.random.default_rng(2)
    p = np.column_stack([rng.uniform(-3e5, 3e5, 20), rng.uniform(-2e5, 2e5, 20), rng.uniform(4e5, 9e5, 20)])
    J = cam.project_jacobian(p)
    h = 1.0
    for k in range(3):
        d = np.zeros(3); d[k] = h
        fd = (cam.project(p + d) - cam.project(p - d)) / (2 * h)
        assert np.abs(J[:, :, k] - fd).max() < 1e-6 * max(1.0, np.abs(fd).max())


def test_no_distortion_is_pinhole_and_fov():
    cam = PinholeCamera.from_fov(hfov_deg=60)
    ray = np.array([np.tan(np.radians(30)), 0.0, 1.0])
    assert cam.project(ray)[0, 0] == pytest.approx(cam.width, abs=1e-9)  # edge of the image
    assert np.allclose(cam.project([0, 0, 5.0]), [[cam.cx, cam.cy]])
    assert np.isnan(cam.project([0, 0, -1.0])).all()


def test_nadir_attitude_boresight_and_orthonormal():
    for e, inc in [(0.001, 51.6), (0.2, 98.0)]:
        x = kepler_to_cartesian(R_EARTH + 600e3, e, np.radians(inc), 0.4, 1.1, 2.0)
        R = NadirAttitude().rotation(0.0, x)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-14) and np.isclose(np.linalg.det(R), 1)
        assert np.allclose(R @ (-x[:3] / np.linalg.norm(x[:3])), [0, 0, 1], atol=1e-14)
        assert (R @ x[3:])[0] > 0 and abs((R @ x[3:])[1]) < 1e-9  # image +x is along-track


def test_nadir_landmark_at_principal_point():
    cam = PinholeCamera.from_fov()
    x = kepler_to_cartesian(R_EARTH + 500e3, 0.0, np.radians(60), 0.0, 0.0, 1.0)
    R = NadirAttitude().rotation(0.0, x)
    nadir_point = x[:3] / np.linalg.norm(x[:3]) * R_EARTH
    uv = cam.project(R @ (nadir_point - x[:3]))
    assert np.allclose(uv, [[cam.cx, cam.cy]], atol=1e-9)


def test_nadir_rotation_jacobian_matches_perturbation():
    x = kepler_to_cartesian(R_EARTH + 600e3, 0.05, np.radians(70), 0.4, 1.1, 2.0)
    att = NadirAttitude(yaw_deg=10)
    dR = att.rotation_jacobian(0.0, x)
    d = np.zeros(6); d[0] = 1e3
    first_order = att.rotation(0.0, x) + 1e3 * dR[0]
    assert np.abs(att.rotation(0.0, x + d) - first_order).max() < 1e-6


def test_telemetry_slerp_reproduces_samples_and_interpolates():
    rots = Rotation.from_euler("z", [[0], [30], [60]], degrees=True)
    att = TelemetryAttitude([0.0, 10.0, 20.0], rots.as_quat())
    for t, r in zip([0.0, 10.0, 20.0], rots):
        assert np.allclose(att.rotation(t), r.as_matrix(), atol=1e-12)
    assert np.allclose(att.rotation(5.0), Rotation.from_euler("z", 15, degrees=True).as_matrix(), atol=1e-12)
    assert np.all(att.rotation_jacobian(0.0) == 0)
