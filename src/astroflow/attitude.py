"""Attitude models: rotation from inertial (ECI) to the camera frame.

`rotation(t, x)` returns R with p_cam = R @ p_eci. Nadir pointing depends on the state, so
`rotation_jacobian` gives dR/dx (6, 3, 3); telemetry attitude does not, so it is zero.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class Attitude(Protocol):
    def rotation(self, t: float, x: np.ndarray) -> np.ndarray: ...
    def rotation_jacobian(self, t: float, x: np.ndarray) -> np.ndarray: ...


class NadirAttitude:
    """Boresight (+z) toward Earth's center; image +x along the velocity component
    perpendicular to the boresight (the yaw rule), optionally rotated by `yaw_deg` about z."""

    def __init__(self, yaw_deg: float = 0.0):
        self.yaw = np.radians(yaw_deg)

    def rotation(self, t, x):
        r, v = np.asarray(x[:3], float), np.asarray(x[3:], float)
        z = -r / np.linalg.norm(r)
        xa = v - (v @ z) * z
        xa /= np.linalg.norm(xa)
        y = np.cross(z, xa)
        R = np.stack([xa, y, z])  # rows = camera axes in ECI
        if self.yaw:
            c, s = np.cos(self.yaw), np.sin(self.yaw)
            R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ R
        return R

    def rotation_jacobian(self, t, x, rel_step: float = 1e-7):
        """Central finite differences of R in r (step ~1 m) and v (step ~1e-3 m/s)."""
        x = np.asarray(x, float)
        out = np.zeros((6, 3, 3))
        for k in range(6):
            h = 1.0 if k < 3 else 1e-3
            d = np.zeros(6); d[k] = h
            out[k] = (self.rotation(t, x + d) - self.rotation(t, x - d)) / (2 * h)
        return out


class TelemetryAttitude:
    """Known attitude from telemetry: quaternions (scalar-last, ECI -> camera) at sample times,
    interpolated with slerp (clamped outside the sampled range)."""

    def __init__(self, times, quats):
        self.times = np.asarray(times, float)
        self._slerp = Slerp(self.times, Rotation.from_quat(np.asarray(quats, float)))

    def rotation(self, t, x=None):
        t = float(np.clip(t, self.times[0], self.times[-1]))
        return self._slerp([t]).as_matrix()[0]

    def rotation_jacobian(self, t, x=None):
        return np.zeros((6, 3, 3))
