import numpy as np
import pytest

from astroflow.constants import MU_EARTH
from astroflow.dynamics import (J2Gravity, Propagator, TwoBody, cartesian_to_kepler,
                                mean_motion, propagate_covariance)

MODELS = [TwoBody(), J2Gravity()]
IDS = ["two_body", "j2"]


def _period(x):
    return 2 * np.pi / mean_motion(cartesian_to_kepler(x)[0])


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_stm_matches_finite_difference(model, x0):
    prop = Propagator(model)
    tf = 1.5 * _period(x0)
    xf, phi = prop.propagate(x0, 0.0, tf)
    fd = np.zeros((6, 6))
    for k in range(6):
        h = 1.0 if k < 3 else 1e-3
        d = np.zeros(6); d[k] = h
        fd[:, k] = (prop.propagate(x0 + d, 0, tf)[0] - prop.propagate(x0 - d, 0, tf)[0]) / (2 * h)
    # Compare in a scale-aware way per column
    for k in range(6):
        assert np.abs(phi[:, k] - fd[:, k]).max() < 1e-5 * np.abs(phi[:, k]).max()


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_stm_structure(model, x0):
    prop = Propagator(model)
    T = _period(x0)
    _, phi0 = prop.propagate(x0, 0.0, 0.0)
    assert np.array_equal(phi0, np.eye(6))
    x1, phi1 = prop.propagate(x0, 0.0, 0.4 * T)
    x2, phi2 = prop.propagate(x0, 0.0, T)
    _, phi21 = prop.propagate(x1, 0.4 * T, T)
    assert np.allclose(phi21 @ phi1, phi2, rtol=1e-7, atol=1e-7 * np.abs(phi2).max())
    J = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])
    assert np.allclose(phi2.T @ J @ phi2, J, atol=1e-6 * np.abs(phi2).max() ** 2)
    assert np.isclose(np.linalg.det(phi2), 1.0, rtol=1e-6)
    xb, _ = prop.propagate(x2, T, 0.0)
    assert np.linalg.norm(xb[:3] - x0[:3]) < 1e-9 * np.linalg.norm(x0[:3])
    assert np.linalg.norm(xb[3:] - x0[3:]) < 1e-9 * np.linalg.norm(x0[3:])


def test_two_body_invariants_and_period_closure(x0):
    prop = Propagator(TwoBody())
    T = _period(x0)
    times = np.linspace(0, 10 * T, 41)
    xs, _ = prop.propagate_many(x0, 0.0, times)
    energy = 0.5 * (xs[:, 3:] ** 2).sum(1) - MU_EARTH / np.linalg.norm(xs[:, :3], axis=1)
    h = np.cross(xs[:, :3], xs[:, 3:])
    assert np.ptp(energy) / abs(energy[0]) < 1e-9
    assert np.abs(h - h[0]).max() / np.linalg.norm(h[0]) < 1e-9
    xT, _ = prop.propagate(x0, 0.0, T)
    assert np.linalg.norm(xT[:3] - x0[:3]) < 1e-3


def test_propagate_many_matches_single_and_handles_backward(x0):
    prop = Propagator(J2Gravity())
    T = _period(x0)
    times = [-0.3 * T, 0.0, 0.2 * T, 0.7 * T]
    xs, phis = prop.propagate_many(x0, 0.0, times)
    for t, x, p in zip(times, xs, phis):
        xs1, p1 = prop.propagate(x0, 0.0, t)
        assert np.allclose(x, xs1, rtol=1e-9, atol=1e-6)
        assert np.allclose(p, p1, rtol=1e-8, atol=1e-8)


def test_covariance_propagation_is_phi_p_phit():
    phi = np.random.default_rng(0).normal(size=(6, 6))
    P0 = np.diag([1, 1, 1, 1e-4, 1e-4, 1e-4])
    Q = 0.1 * np.eye(6)
    assert np.allclose(propagate_covariance(P0, phi), phi @ P0 @ phi.T)
    assert np.allclose(propagate_covariance(P0, phi, Q), phi @ P0 @ phi.T + Q)
