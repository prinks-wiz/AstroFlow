"""Measurement model: predicted pixels (and Jacobians) for landmarks seen from an orbit.

Chain: state at epoch --STM/propagation--> state at t_k --> landmark in ECI (Earth rotation)
--> line of sight --> camera frame (attitude) --> pixels (camera).
The same function generates simulated data and serves as the estimators' model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attitude import Attitude
from .camera import Camera
from .dynamics import Propagator
from .frames import EarthRotation


@dataclass
class Prediction:
    pixels: np.ndarray      # (N, 2); nan where the landmark is behind the camera
    valid: np.ndarray       # (N,) bool: in front of the camera
    range_m: np.ndarray     # (N,) distance satellite -> landmark
    jacobian: np.ndarray | None  # (N, 2, 6) d(pixel)/d(state at epoch), or None


def predict_pixels(x_ref, t_ref, t, landmarks_ecef, propagator: Propagator, earth: EarthRotation,
                   attitude: Attitude, camera: Camera, jacobian: bool = True,
                   states_cache=None) -> Prediction:
    """Predict pixels for landmark k seen at time t[k].

    `states_cache` may be a precomputed (times, states, stms) triple from
    `propagator.propagate_many` over the unique times, to avoid re-propagating.
    """
    t = np.asarray(t, float)
    lm = np.asarray(landmarks_ecef, float).reshape(-1, 3)
    uniq, inverse = np.unique(t, return_inverse=True)
    if states_cache is None:
        states, phis = propagator.propagate_many(x_ref, t_ref, uniq)
    else:
        uniq_c, states, phis = states_cache
        assert np.array_equal(uniq_c, uniq)

    n = len(t)
    uv = np.full((n, 2), np.nan)
    valid = np.zeros(n, bool)
    rng = np.zeros(n)
    jac = np.zeros((n, 2, 6)) if jacobian else None

    for g, tk in enumerate(uniq):
        idx = np.where(inverse == g)[0]
        xk, phi = states[g], phis[g]
        rk = xk[:3]
        l_eci = lm[idx] @ earth.rotation(tk)  # rows: R^T l  (r_eci = R^T r_ecef)
        rel = l_eci - rk
        R = attitude.rotation(tk, xk)
        p_cam = rel @ R.T
        uv[idx] = camera.project(p_cam)
        valid[idx] = p_cam[:, 2] > 0
        rng[idx] = np.linalg.norm(rel, axis=1)
        if jacobian:
            Jp = camera.project_jacobian(p_cam)                 # (m, 2, 3)
            dR = attitude.rotation_jacobian(tk, xk)             # (6, 3, 3)
            dp = np.zeros((len(idx), 3, 6))
            dp[:, :, :3] = -R                                   # d p_cam / d r_k
            dp += np.einsum("jab,mb->maj", dR, rel)             # attitude dependence on the state
            jac[idx] = Jp @ dp @ phi
    return Prediction(uv, valid, rng, jac)
