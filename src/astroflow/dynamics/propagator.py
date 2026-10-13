"""Orbit propagation with state transition matrix (STM)."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .force_models import ForceModel


class Propagator:
    """Propagates x = [r; v] and the STM Phi(t, t0) via the variational equations."""

    def __init__(self, force_model: ForceModel, rtol: float = 1e-12, atol: float = 1e-9):
        self.force_model = force_model
        self.rtol, self.atol = rtol, atol

    def _rhs(self, t, y):
        r, v = y[:3], y[3:6]
        phi = y[6:].reshape(6, 6)
        da_dr, da_dv = self.force_model.jacobian(t, r, v)
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = da_dr
        A[3:, 3:] = da_dv
        dy = np.empty(42)
        dy[:3] = v
        dy[3:6] = self.force_model.accel(t, r, v)
        dy[6:] = (A @ phi).ravel()
        return dy

    def propagate_many(self, x0, t0: float, times):
        """Return states (N, 6) and STMs (N, 6, 6) at `times` (any order relative to t0
        but monotonic; times may be before t0 for backward propagation)."""
        times = np.atleast_1d(np.asarray(times, dtype=float))
        y0 = np.concatenate([np.asarray(x0, dtype=float), np.eye(6).ravel()])
        states = np.empty((len(times), 6))
        phis = np.empty((len(times), 6, 6))
        for sign in (1.0, -1.0):
            idx = np.where(sign * (times - t0) > 0)[0]
            if len(idx):
                order = idx[np.argsort(sign * times[idx])]
                sol = solve_ivp(self._rhs, (t0, times[order[-1]]), y0, method="DOP853",
                                t_eval=times[order], rtol=self.rtol, atol=self.atol)
                if not sol.success:
                    raise RuntimeError(sol.message)
                states[order] = sol.y[:6].T
                phis[order] = sol.y[6:].T.reshape(-1, 6, 6)
        for i in np.where(times == t0)[0]:
            states[i], phis[i] = x0, np.eye(6)
        return states, phis

    def propagate(self, x0, t0: float, tf: float):
        """Return state (6,) and STM Phi(tf, t0) (6, 6)."""
        s, p = self.propagate_many(x0, t0, [tf])
        return s[0], p[0]


def propagate_covariance(P0, phi, Q=None):
    """P(t) = Phi P0 Phi^T (+ Q)."""
    P = phi @ P0 @ phi.T
    return P if Q is None else P + Q
