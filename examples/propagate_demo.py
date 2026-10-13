"""Propagate a LEO orbit and report energy drift and J2 nodal regression."""
import numpy as np

from astroflow.constants import MU_EARTH, R_EARTH
from astroflow.dynamics import (J2Gravity, Propagator, TwoBody, cartesian_to_kepler,
                                kepler_to_cartesian, mean_motion)

a = R_EARTH + 700e3
x0 = kepler_to_cartesian(a, 0.001, np.radians(98.2), 0.5, 1.0, 0.0)
T = 2 * np.pi / mean_motion(a)
times = np.linspace(0, 100 * T, 2001)

xs, _ = Propagator(TwoBody()).propagate_many(x0, 0.0, times)
E = 0.5 * (xs[:, 3:] ** 2).sum(1) - MU_EARTH / np.linalg.norm(xs[:, :3], axis=1)
print(f"two-body energy drift over 100 orbits: {np.ptp(E) / abs(E[0]):.2e} (relative)")

xs, _ = Propagator(J2Gravity()).propagate_many(x0, 0.0, times)
raan = np.unwrap([cartesian_to_kepler(x)[3] for x in xs])
rate = np.degrees(np.polyfit(times, raan, 1)[0]) * 86400
print(f"J2 nodal regression: {rate:.4f} deg/day (sun-synchronous target ~0.9856)")
