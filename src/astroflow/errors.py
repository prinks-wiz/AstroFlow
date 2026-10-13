"""Observation container and pixel-error model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Observations:
    """What an orbit-determination estimator sees; the front end's whole contribution.

    t:             (N,) frame timestamp (s since epoch) for each point
    pixels:        (N, 2) measured pixel coordinates (u, v)
    landmark_ecef: (N, 3) matched landmark position as reported by the landmark database
    sigma_px:      (N,) 1-sigma pixel uncertainty per point (carries match confidence)
    """
    t: np.ndarray
    pixels: np.ndarray
    landmark_ecef: np.ndarray
    sigma_px: np.ndarray

    def __post_init__(self):
        self.t = np.asarray(self.t, float)
        self.pixels = np.asarray(self.pixels, float).reshape(-1, 2)
        self.landmark_ecef = np.asarray(self.landmark_ecef, float).reshape(-1, 3)
        self.sigma_px = np.asarray(self.sigma_px, float)
        n = len(self.t)
        assert self.pixels.shape == (n, 2) and self.landmark_ecef.shape == (n, 3) and self.sigma_px.shape == (n,)

    def __len__(self):
        return len(self.t)

    def subset(self, mask) -> "Observations":
        return Observations(self.t[mask], self.pixels[mask], self.landmark_ecef[mask], self.sigma_px[mask])


def landmark_sigma_to_pixels(sigma_m, range_m, focal_px):
    """A landmark position error of sigma_m (per axis, perpendicular to the line of sight)
    seen from range_m shifts the projected pixel by about focal_px * sigma_m / range_m."""
    return focal_px * np.asarray(sigma_m, float) / np.asarray(range_m, float)


def total_pixel_sigma(pixel_sigma, landmark_sigma_m, range_m, focal_px):
    """Per-axis pixel sigma combining image noise and landmark geolocation error."""
    return np.hypot(pixel_sigma, landmark_sigma_to_pixels(landmark_sigma_m, range_m, focal_px))
