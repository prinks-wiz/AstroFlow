"""Camera settings and a MODELED effect of exposure / gain / white balance on match quality.

IMPORTANT: everything in `MatchQualityModel` except the motion-blur geometry is a modeling
assumption with explicit, configurable coefficients. It is NOT measured Arducam behavior.
It exists so the effect of camera settings on orbit determination can be explored before the
image-level sensor model (stage B) exists; calibrate it against rendered or flight imagery later.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import R_EARTH


@dataclass(frozen=True)
class CameraSettings:
    exposure_s: float = 0.005
    gain: float = 1.0
    wb_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)  # (R, G, B) multipliers applied by the camera


@dataclass(frozen=True)
class QualityOutput:
    sigma_px: np.ndarray       # pixel sigma after blur / noise effects
    p_detect: np.ndarray       # probability a landmark is detected and matched
    p_outlier_extra: np.ndarray  # additional mismatch probability (white balance mismatch)
    blur_px: np.ndarray
    snr: np.ndarray
    saturated: np.ndarray      # fraction-of-range proxy: gain * signal / full scale (>1 saturates)


@dataclass
class MatchQualityModel:
    """Maps (settings, scene) -> measurement quality. All coefficients are assumptions.

    signal electrons   N = n_ref * (exposure / exposure_ref) * irradiance   (irradiance = sin(sun elev), 0 at night)
    saturation level   S = gain * N / n_sat ; the signal clips at S = 1
    noise              sqrt(N + read_e^2 + (q_e / gain)^2)  -> SNR = N / noise
    blur               blur_px = ground_speed * exposure / GSD   (pure geometry)
    pixel sigma        sqrt(sigma0^2 + (blur_px / sqrt(12))^2 + (loc_k / SNR)^2)
    p_detect           p_max * (1 - exp(-(SNR/snr_half)^2)) * exp(-(blur_px/blur_half)^2) * sat_factor
    extra outliers     k_wb * wb_cast^2  (wb_cast = distance of the WB gains from the reference balance, in log space)
    """
    enabled: bool = True
    n_ref: float = 3000.0          # electrons at exposure_ref under full sun with albedo folded in
    exposure_ref: float = 0.005
    n_sat: float = 20000.0         # full well (electrons)
    read_e: float = 4.0
    q_e: float = 12.0              # ADC step in electrons at gain 1
    loc_k: float = 0.5             # px of localization error at SNR = 1
    snr_half: float = 4.0
    blur_half: float = 6.0         # px of blur at which detection falls to ~37%
    p_max: float = 0.98
    k_wb: float = 0.6
    ref_wb_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)  # balance of the reference imagery
    mu_ground: float = R_EARTH     # for ground speed = |v| * R / |r|

    @staticmethod
    def disabled() -> "MatchQualityModel":
        return MatchQualityModel(enabled=False)

    def wb_cast(self, wb_gains) -> float:
        g = np.asarray(wb_gains, float) / np.asarray(self.ref_wb_gains, float)
        g = g / g[1]  # relative to green
        return float(np.hypot(np.log(g[0]), np.log(g[2])))

    def evaluate(self, settings: CameraSettings, range_m, speed_mps, radius_m, sun_elev_rad,
                 focal_px, base_sigma_px) -> QualityOutput:
        range_m = np.atleast_1d(np.asarray(range_m, float))
        sun_elev_rad = np.broadcast_to(np.asarray(sun_elev_rad, float), range_m.shape)
        base = np.broadcast_to(np.asarray(base_sigma_px, float), range_m.shape).copy()
        gsd = range_m / focal_px                                  # m per pixel (nadir approximation)
        ground_speed = speed_mps * self.mu_ground / radius_m
        blur = ground_speed * settings.exposure_s / gsd
        if not self.enabled:
            ones = np.ones_like(range_m)
            return QualityOutput(base, ones, np.zeros_like(range_m), blur, np.full_like(range_m, np.inf),
                                 np.zeros_like(range_m))
        irr = np.clip(np.sin(sun_elev_rad), 0.0, None)
        N = self.n_ref * (settings.exposure_s / self.exposure_ref) * irr
        sat = settings.gain * N / self.n_sat
        noise = np.sqrt(N + self.read_e**2 + (self.q_e / settings.gain) ** 2)
        snr = N / noise
        # clipping destroys contrast: detection falls once the signal passes full scale
        sat_factor = np.where(sat <= 1.0, 1.0, np.exp(-2.0 * (sat - 1.0)))
        eff_snr = np.minimum(snr, snr * 1.0 / np.maximum(sat, 1.0))
        sigma = np.sqrt(base**2 + (blur / np.sqrt(12.0)) ** 2 + (self.loc_k / np.maximum(eff_snr, 1e-9)) ** 2)
        p_det = (self.p_max * (1.0 - np.exp(-(snr / self.snr_half) ** 2))
                 * np.exp(-((blur / self.blur_half) ** 2)) * sat_factor)
        p_out = np.full_like(range_m, min(1.0, self.k_wb * self.wb_cast(settings.wb_gains) ** 2))
        return QualityOutput(sigma, p_det, p_out, blur, snr, sat)
