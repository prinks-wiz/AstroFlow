"""Pluggable force models returning acceleration and its analytic Jacobians."""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..constants import J2_EARTH, MU_EARTH, R_EARTH


class ForceModel(Protocol):
    def accel(self, t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Inertial acceleration (3,) in m/s^2."""

    def jacobian(self, t: float, r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(da/dr, da/dv), each (3, 3)."""


class TwoBody:
    def __init__(self, mu: float = MU_EARTH):
        self.mu = mu

    def accel(self, t, r, v):
        return -self.mu * r / np.linalg.norm(r) ** 3

    def jacobian(self, t, r, v):
        rn = np.linalg.norm(r)
        da_dr = 3.0 * self.mu * np.outer(r, r) / rn**5 - self.mu * np.eye(3) / rn**3
        return da_dr, np.zeros((3, 3))


class J2Gravity:
    """Two-body plus the J2 zonal harmonic (Earth's pole along inertial z)."""

    def __init__(self, mu: float = MU_EARTH, j2: float = J2_EARTH, radius: float = R_EARTH):
        self.mu, self.j2, self.radius = mu, j2, radius
        self._two_body = TwoBody(mu)

    def _k(self):
        return 1.5 * self.j2 * self.mu * self.radius**2

    def accel(self, t, r, v):
        x, y, z = r
        rn = np.linalg.norm(r)
        s = 5.0 * z * z / rn**2
        a_j2 = -self._k() / rn**5 * np.array([x * (1 - s), y * (1 - s), z * (3 - s)])
        return self._two_body.accel(t, r, v) + a_j2

    def jacobian(self, t, r, v):
        x, y, z = r
        rn = np.linalg.norm(r)
        k = self._k()
        z2 = z * z
        r2 = rn * rn
        # a_j2 = -k * g / r^5, g = [x(1-s), y(1-s), z(3-s)], s = 5 z^2 / r^2
        s = 5.0 * z2 / r2
        g = np.array([x * (1 - s), y * (1 - s), z * (3 - s)])
        ds_dr = np.array([-10.0 * z2 * x / r2**2, -10.0 * z2 * y / r2**2,
                          10.0 * z / r2 - 10.0 * z2 * z / r2**2])
        dg = np.zeros((3, 3))
        dg[0] = -x * ds_dr; dg[0, 0] += 1 - s
        dg[1] = -y * ds_dr; dg[1, 1] += 1 - s
        dg[2] = -z * ds_dr; dg[2, 2] += 3 - s
        # d(r^-5)/dr = -5 r^-7 * r_vec
        da = -k * (dg / rn**5 - 5.0 * np.outer(g, r) / rn**7)
        da_tb, _ = self._two_body.jacobian(t, r, v)
        return da_tb + da, np.zeros((3, 3))


class CompositeForce:
    """Sum of force models (e.g. a gravity model plus drag)."""

    def __init__(self, models: Sequence[ForceModel]):
        self.models = list(models)

    def accel(self, t, r, v):
        return sum(m.accel(t, r, v) for m in self.models)

    def jacobian(self, t, r, v):
        pairs = [m.jacobian(t, r, v) for m in self.models]
        return sum(p[0] for p in pairs), sum(p[1] for p in pairs)
