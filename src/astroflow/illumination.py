"""Low-precision sun model, used only to decide how brightly the ground is lit."""
from __future__ import annotations

import numpy as np

from .constants import OBLIQUITY_RAD, YEAR_S


class SimpleSun:
    """Sun on a circular ecliptic orbit. `lambda0` is the ecliptic longitude at t = 0
    (0 = March equinox, pi/2 = June solstice)."""

    def __init__(self, lambda0: float = 0.0):
        self.lambda0 = lambda0

    @classmethod
    def towards(cls, r_eci, offset: float = 0.0) -> "SimpleSun":
        """Sun placed as close as the ecliptic allows to the direction of r_eci at t = 0
        (so that point is near local noon); `offset` (rad) rotates it along the ecliptic."""
        e = OBLIQUITY_RAD
        r = np.asarray(r_eci, float)
        return cls(np.arctan2(r[1] * np.cos(e) + r[2] * np.sin(e), r[0]) + offset)

    def direction_eci(self, t: float) -> np.ndarray:
        lam = self.lambda0 + 2 * np.pi * t / YEAR_S
        e = OBLIQUITY_RAD
        return np.array([np.cos(lam), np.sin(lam) * np.cos(e), np.sin(lam) * np.sin(e)])


def sun_elevation(sun_dir_eci, ground_up_eci):
    """Sun elevation angle (rad) above the local horizon, for unit 'up' vectors (N, 3)."""
    s = np.clip(np.atleast_2d(ground_up_eci) @ np.asarray(sun_dir_eci, float), -1.0, 1.0)
    return np.arcsin(s)
