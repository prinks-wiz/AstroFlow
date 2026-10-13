import numpy as np
import pytest

from astroflow.constants import A_WGS84, B_WGS84, OMEGA_EARTH
from astroflow.frames import SimpleEarthRotation, ecef_to_eci, eci_to_ecef
from astroflow.geodesy import (ecef_to_geodetic, geodetic_to_ecef, ray_ellipsoid, sees_point,
                               surface_normal)


def test_eci_ecef_roundtrip_and_orthonormal():
    earth = SimpleEarthRotation(theta0=0.3)
    r = np.array([7e6, -1e6, 2e6])
    for t in (0.0, 1234.5, 86400.0):
        R = earth.rotation(t)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-14) and np.isclose(np.linalg.det(R), 1)
        assert np.allclose(ecef_to_eci(earth, t, eci_to_ecef(earth, t, r)), r, atol=1e-8)


def test_earth_rotates_eastward_at_sidereal_rate():
    earth = SimpleEarthRotation()
    r_eci = np.array([A_WGS84, 0.0, 0.0])  # on the equator at lon 0 at t=0
    dt = 600.0
    lon = np.arctan2(*eci_to_ecef(earth, dt, r_eci)[[1, 0]])
    assert lon == pytest.approx(-OMEGA_EARTH * dt)  # fixed inertial point drifts west in ECEF


def test_equator_and_pole_known_values():
    assert np.allclose(geodetic_to_ecef(0, 0, 0), [A_WGS84, 0, 0])
    assert np.allclose(geodetic_to_ecef(np.pi / 2, 0, 0), [0, 0, B_WGS84], atol=1e-6)


def test_geodetic_roundtrip():
    rng = np.random.default_rng(1)
    lat = rng.uniform(-np.pi / 2, np.pi / 2, 500)
    lon = rng.uniform(-np.pi, np.pi, 500)
    h = rng.uniform(-1e3, 1e6, 500)
    lat2, lon2, h2 = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
    assert np.degrees(np.abs(lat2 - lat)).max() < 1e-9
    assert np.degrees(np.abs(lon2 - lon)).max() < 1e-9
    assert np.abs(h2 - h).max() < 1e-5


def test_ray_ellipsoid_hits_surface_and_misses_sky():
    sat = geodetic_to_ecef(0.2, 0.4, 500e3)
    nadir = -surface_normal(0.2, 0.4)
    pts, hit = ray_ellipsoid(sat, [nadir, -nadir])
    assert hit.tolist() == [True, False]
    assert np.allclose(pts[0], geodetic_to_ecef(0.2, 0.4, 0.0), atol=1e-6)
    lat, lon, h = ecef_to_geodetic(pts[0])
    assert abs(h) < 1e-6


def test_horizon_test():
    sat = geodetic_to_ecef(0.0, 0.0, 500e3)
    near = geodetic_to_ecef(0.0, np.radians(5), 0.0)
    far_side = geodetic_to_ecef(0.0, np.radians(90), 0.0)
    assert sees_point(sat, near)[0] and not sees_point(sat, far_side)[0]
