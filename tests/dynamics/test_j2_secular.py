import numpy as np
import pytest

from astroflow.constants import J2_EARTH, R_EARTH
from astroflow.dynamics import J2Gravity, Propagator, cartesian_to_kepler, kepler_to_cartesian, mean_motion


def _rates(a, e, i_deg, n_orbits=60):
    i = np.radians(i_deg)
    x0 = kepler_to_cartesian(a, e, i, 0.5, 1.0, 0.0)
    n = mean_motion(a)
    T = 2 * np.pi / n
    times = np.linspace(0, n_orbits * T, n_orbits * 20 + 1)
    xs, _ = Propagator(J2Gravity()).propagate_many(x0, 0.0, times)
    el = np.array([cartesian_to_kepler(x) for x in xs])
    raan, argp = np.unwrap(el[:, 3]), np.unwrap(el[:, 4])
    p = a * (1 - e**2)
    fac = 1.5 * n * J2_EARTH * (R_EARTH / p) ** 2
    theory = (-fac * np.cos(i), 0.5 * fac * (5 * np.cos(i) ** 2 - 1))
    return (np.polyfit(times, raan, 1)[0], np.polyfit(times, argp, 1)[0]), theory


@pytest.mark.parametrize("a,e,inc", [(R_EARTH + 500e3, 0.01, 51.6), (R_EARTH + 1200e3, 0.05, 30.0),
                                     (R_EARTH + 800e3, 0.02, 98.0), (26000e3, 0.4, 45.0)])
def test_secular_rates_match_theory(a, e, inc):
    (raan_dot, argp_dot), (raan_th, argp_th) = _rates(a, e, inc)
    assert raan_dot == pytest.approx(raan_th, rel=0.02)
    assert argp_dot == pytest.approx(argp_th, rel=0.02)


def test_sun_synchronous():
    a = R_EARTH + 700e3
    # solve for the sun-synchronous inclination
    n = mean_motion(a)
    target = np.radians(0.9856) / 86400
    cosi = -target / (1.5 * n * J2_EARTH * (R_EARTH / a) ** 2)
    (raan_dot, _), _ = _rates(a, 0.001, np.degrees(np.arccos(cosi)))
    assert np.degrees(raan_dot) * 86400 == pytest.approx(0.9856, rel=0.02)


def test_critical_inclination_freezes_perigee():
    (_, argp_dot), _ = _rates(R_EARTH + 1500e3, 0.05, np.degrees(np.arctan(2)))
    n = mean_motion(R_EARTH + 1500e3)
    assert abs(argp_dot) < 0.02 * 1.5 * n * J2_EARTH * (R_EARTH / (R_EARTH + 1500e3)) ** 2
