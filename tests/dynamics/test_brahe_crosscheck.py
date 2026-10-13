"""Cross-check against Brahe's numerical propagator (tight tolerances, rotation-only frame
so the J2 pole is inertial +z as in our model).

Brahe's built-in STM is not used as a reference: it differs from finite differences of
Brahe's own propagated states by ~2%, while ours agrees to ~1e-9.
"""
import numpy as np
import pytest

bh = pytest.importorskip("brahe")

from astroflow.dynamics import J2Gravity, Propagator, TwoBody, kepler_to_cartesian, mean_motion

A = bh.R_EARTH + 500e3
X0 = kepler_to_cartesian(A, 0.01, np.radians(51.6), 0.5, 1.0, 0.3, mu=bh.GM_EARTH)
T = 2 * np.pi / mean_motion(A, bh.GM_EARTH)

GRAVITY_J2 = bh.GravityConfiguration(degree=2, order=0, model_type=bh.GravityModelType.EGM2008_120)
CASES = {
    "two_body": (bh.ForceModelConfig.two_body(), TwoBody(bh.GM_EARTH)),
    "j2": (bh.ForceModelConfig(gravity=GRAVITY_J2,
                               frame_transform=bh.FrameTransformationModel.EARTH_ROTATION_ONLY),
           J2Gravity(bh.GM_EARTH, bh.J2_EARTH, bh.R_EARTH)),
}


def _brahe_state(cfg, x0, dur):
    bh.initialize_eop()
    ep = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    p = bh.NumericalOrbitPropagator(ep, x0, bh.NumericalPropagationConfig.high_precision(), cfg, None)
    p.propagate_to(ep + dur)
    return np.array(p.current_state())


@pytest.mark.parametrize("case", CASES)
def test_state_matches_brahe(case):
    cfg, model = CASES[case]
    xb = _brahe_state(cfg, X0, 3 * T)
    xm, _ = Propagator(model).propagate(X0, 0.0, 3 * T)
    assert np.linalg.norm(xb[:3] - xm[:3]) < 1e-3  # measured ~3e-5 m
    assert np.linalg.norm(xb[3:] - xm[3:]) < 1e-6


@pytest.mark.parametrize("case", CASES)
def test_stm_matches_finite_difference_of_brahe(case):
    cfg, model = CASES[case]
    dur = 0.5 * T
    _, phi = Propagator(model).propagate(X0, 0.0, dur)
    fd = np.zeros((6, 6))
    for k in range(6):
        h = 1.0 if k < 3 else 1e-3
        d = np.zeros(6); d[k] = h
        fd[:, k] = (_brahe_state(cfg, X0 + d, dur) - _brahe_state(cfg, X0 - d, dur)) / (2 * h)
    assert np.abs(phi - fd).max() < 1e-5 * np.abs(phi).max()
