"""Keplerian <-> Cartesian conversions (inertial frame, angles in radians)."""
from __future__ import annotations

import numpy as np

from ..constants import MU_EARTH


def mean_motion(a: float, mu: float = MU_EARTH) -> float:
    return np.sqrt(mu / a**3)


def kepler_to_cartesian(a, e, i, raan, argp, nu, mu: float = MU_EARTH):
    p = a * (1 - e**2)
    r_pf = p / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    def rot(ang, axis):
        c, s = np.cos(ang), np.sin(ang)
        if axis == "z":
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    Q = rot(raan, "z") @ rot(i, "x") @ rot(argp, "z")
    return np.concatenate([Q @ r_pf, Q @ v_pf])


def cartesian_to_kepler(x, mu: float = MU_EARTH):
    """Return (a, e, i, raan, argp, nu). Assumes non-circular, non-equatorial orbit."""
    r, v = np.asarray(x[:3], float), np.asarray(x[3:], float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    n_vec = np.array([-h[1], h[0], 0.0])
    e_vec = np.cross(v, h) / mu - r / rn
    e = np.linalg.norm(e_vec)
    a = 1.0 / (2.0 / rn - vn**2 / mu)
    i = np.arccos(h[2] / np.linalg.norm(h))
    nn = np.linalg.norm(n_vec)
    raan = np.arctan2(n_vec[1], n_vec[0]) % (2 * np.pi)
    argp = np.arccos(np.clip(n_vec @ e_vec / (nn * e), -1, 1))
    if e_vec[2] < 0:
        argp = 2 * np.pi - argp
    nu = np.arccos(np.clip(e_vec @ r / (e * rn), -1, 1))
    if r @ v < 0:
        nu = 2 * np.pi - nu
    return a, e, i, raan, argp, nu
