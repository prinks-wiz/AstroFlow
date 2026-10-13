"""ECI <-> ECEF. Components are swappable: anything with `rotation(t)` works."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .constants import OMEGA_EARTH


class EarthRotation(Protocol):
    def rotation(self, t: float) -> np.ndarray:
        """3x3 matrix R with r_ecef = R @ r_eci at time t (seconds since epoch)."""


class SimpleEarthRotation:
    """Pure rotation about +z at the sidereal rate (no precession/nutation/polar motion).

    This matches the `EARTH_ROTATION_ONLY` frame used in the Brahe cross-check.
    `theta0` is the Greenwich angle at t = 0.
    """

    def __init__(self, theta0: float = 0.0, omega: float = OMEGA_EARTH):
        self.theta0, self.omega = theta0, omega

    def rotation(self, t: float) -> np.ndarray:
        th = self.theta0 + self.omega * t
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def eci_to_ecef(earth: EarthRotation, t: float, r_eci) -> np.ndarray:
    return earth.rotation(t) @ np.asarray(r_eci, float)


def ecef_to_eci(earth: EarthRotation, t: float, r_ecef) -> np.ndarray:
    return earth.rotation(t).T @ np.asarray(r_ecef, float)
