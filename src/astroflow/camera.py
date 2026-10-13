"""Camera geometry: pinhole intrinsics with optional radial distortion.

Camera frame: +z along the boresight, +x to image right, +y to image down.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Camera(Protocol):
    width: int
    height: int

    def project(self, p_cam: np.ndarray) -> np.ndarray: ...
    def project_jacobian(self, p_cam: np.ndarray) -> np.ndarray: ...
    def unproject(self, uv: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0

    @classmethod
    def from_fov(cls, width: int = 1920, height: int = 1080, hfov_deg: float = 60.0, **kw):
        """Square pixels, principal point at the image center (placeholder Arducam-like default)."""
        f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        return cls(f, f, width / 2.0, height / 2.0, width, height, **kw)

    @property
    def focal_px(self) -> float:
        return 0.5 * (self.fx + self.fy)

    def _distort(self, xn, yn):
        r2 = xn * xn + yn * yn
        s = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return xn * s, yn * s

    def project(self, p_cam):
        """(N, 3) or (3,) camera-frame vectors to pixels (N, 2). Behind-camera points give nan."""
        p = np.atleast_2d(np.asarray(p_cam, float))
        z = p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn, yn = p[:, 0] / z, p[:, 1] / z
        xd, yd = self._distort(xn, yn)
        uv = np.stack([self.fx * xd + self.cx, self.fy * yd + self.cy], axis=1)
        uv[z <= 0] = np.nan
        return uv

    def project_jacobian(self, p_cam):
        """d(pixel)/d(p_cam), shape (N, 2, 3)."""
        p = np.atleast_2d(np.asarray(p_cam, float))
        X, Y, Z = p[:, 0], p[:, 1], p[:, 2]
        xn, yn = X / Z, Y / Z
        r2 = xn * xn + yn * yn
        s = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        ds = self.k1 + 2.0 * self.k2 * r2  # ds/d(r2)
        # d(xd, yd)/d(xn, yn)
        dxx, dxy = s + 2 * xn * xn * ds, 2 * xn * yn * ds
        dyx, dyy = 2 * xn * yn * ds, s + 2 * yn * yn * ds
        # d(xn, yn)/d(p)
        J = np.zeros((len(p), 2, 3))
        dn = np.zeros((len(p), 2, 3))
        dn[:, 0, 0] = 1 / Z; dn[:, 0, 2] = -X / Z**2
        dn[:, 1, 1] = 1 / Z; dn[:, 1, 2] = -Y / Z**2
        dd = np.stack([np.stack([dxx, dxy], 1), np.stack([dyx, dyy], 1)], 1)  # (N, 2, 2)
        J = dd @ dn
        J[:, 0] *= self.fx
        J[:, 1] *= self.fy
        return J

    def unproject(self, uv):
        """Pixels (N, 2) to unit rays (N, 3) in the camera frame (inverts distortion by iteration)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        xd, yd = (uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy
        xn, yn = xd.copy(), yd.copy()
        for _ in range(20):
            xe, ye = self._distort(xn, yn)
            xn, yn = xn - (xe - xd), yn - (ye - yd)
        ray = np.stack([xn, yn, np.ones_like(xn)], axis=1)
        return ray / np.linalg.norm(ray, axis=1, keepdims=True)

    def in_image(self, uv, margin: float = 0.0):
        uv = np.atleast_2d(uv)
        return ((uv[:, 0] >= margin) & (uv[:, 0] <= self.width - margin)
                & (uv[:, 1] >= margin) & (uv[:, 1] <= self.height - margin))
