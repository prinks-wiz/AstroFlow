import numpy as np
import pytest

from astroflow.dynamics import CompositeForce, J2Gravity, TwoBody


@pytest.mark.parametrize("model", [TwoBody(), J2Gravity(), CompositeForce([TwoBody(), J2Gravity()])])
def test_jacobian_matches_finite_difference(model, x0):
    r, v = x0[:3], x0[3:]
    da_dr, da_dv = model.jacobian(0.0, r, v)
    h = 1.0
    fd = np.zeros((3, 3))
    for k in range(3):
        d = np.zeros(3); d[k] = h
        fd[:, k] = (model.accel(0, r + d, v) - model.accel(0, r - d, v)) / (2 * h)
    scale = np.abs(da_dr).max()
    assert np.abs(da_dr - fd).max() < 1e-6 * scale
    assert np.all(da_dv == 0)


def test_j2_reduces_to_two_body_when_j2_zero(x0):
    a1 = J2Gravity(j2=0.0).accel(0, x0[:3], x0[3:])
    a2 = TwoBody().accel(0, x0[:3], x0[3:])
    assert np.allclose(a1, a2, rtol=1e-14, atol=0)


def test_composite_sums():
    r, v = np.array([7e6, 1e5, 2e5]), np.zeros(3)
    c = CompositeForce([TwoBody(), TwoBody()])
    assert np.allclose(c.accel(0, r, v), 2 * TwoBody().accel(0, r, v))
