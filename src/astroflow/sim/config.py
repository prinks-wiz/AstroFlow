from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..attitude import Attitude, NadirAttitude
from ..camera import Camera, PinholeCamera
from ..constants import R_EARTH
from ..dynamics import ForceModel, J2Gravity
from ..frames import EarthRotation, SimpleEarthRotation
from ..illumination import SimpleSun
from ..sensor import CameraSettings, MatchQualityModel


@dataclass
class SimConfig:
    # --- orbit (Keplerian at t = 0, radians) and truth dynamics
    orbit: tuple = (R_EARTH + 550e3, 0.002, np.radians(60.0), 0.3, 1.0, 0.5)
    truth_force: ForceModel = field(default_factory=J2Gravity)
    # --- frames
    n_frames: int = 12
    frame_dt: float = 60.0
    # --- points
    inliers_per_frame: int = 20      # attempted inliers per frame (before detection thinning)
    outlier_fraction: float = 0.0    # fraction of each frame's points that are outliers
    outlier_mix: tuple = (0.5, 0.5)  # weights: (random_pixel, wrong_landmark)
    wrong_landmark_offset_m: tuple = (1e3, 1e5)  # log-uniform displacement of near-miss mismatches
    # --- noise
    pixel_sigma: float = 0.5
    landmark_sigma_m: float = 10.0
    sigma_jitter: float = 0.0        # lognormal spread of per-point pixel sigma (match confidence)
    outlier_sigma_inflate: float = 1.0  # reported-sigma multiplier for outliers (1 = indistinguishable)
    # --- camera, settings and quality model
    camera: Camera = field(default_factory=PinholeCamera.from_fov)
    settings: CameraSettings | Sequence[CameraSettings] = field(default_factory=CameraSettings)
    quality: MatchQualityModel = field(default_factory=MatchQualityModel.disabled)
    sun: SimpleSun | None = None  # None: lit so the sub-satellite point at t=0 is near local noon
    # --- other components
    attitude: Attitude = field(default_factory=NadirAttitude)
    earth: EarthRotation = field(default_factory=SimpleEarthRotation)
    seed: int = 0

    def settings_at(self, k: int) -> CameraSettings:
        s = self.settings
        return s if isinstance(s, CameraSettings) else s[min(k, len(s) - 1)]
