from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..dynamics import Propagator, kepler_to_cartesian
from ..errors import Observations, landmark_sigma_to_pixels
from ..illumination import SimpleSun
from ..geodesy import ray_ellipsoid, sees_point, surface_normal, ecef_to_geodetic
from ..measurement import predict_pixels
from . import outliers as ol
from .config import SimConfig


@dataclass
class SimulatedData:
    obs: Observations
    truth_x_epoch: np.ndarray        # (6,) state at t = 0
    frame_times: np.ndarray          # (n_frames,)
    truth_states: np.ndarray         # (n_frames, 6)
    is_outlier: np.ndarray           # (N,) bool -- for scoring only, hidden from estimators
    frame_index: np.ndarray          # (N,) int
    true_landmark_ecef: np.ndarray   # (N, 3) where the pixel truly points
    true_pixels: np.ndarray          # (N, 2) noise-free pixels of the true landmark
    quality: list = field(default_factory=list)  # per-frame QualityOutput summaries
    config: SimConfig | None = None


class ObservationSource(Protocol):
    """Anything that yields observations: this simulator now, a renderer + matcher later."""
    def generate(self, rng: np.random.Generator) -> SimulatedData: ...


class Simulator:
    def __init__(self, config: SimConfig):
        self.cfg = config

    def generate(self, rng: np.random.Generator | None = None) -> SimulatedData:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed) if rng is None else rng
        cam = cfg.camera
        x0 = kepler_to_cartesian(*cfg.orbit)
        sun = cfg.sun or SimpleSun.towards(x0[:3])
        times = np.arange(cfg.n_frames) * cfg.frame_dt
        truth_prop = Propagator(cfg.truth_force)
        states, _ = truth_prop.propagate_many(x0, 0.0, times)

        rows = dict(t=[], uv=[], lm_rep=[], sig=[], out=[], frame=[], lm_true=[], uv_true=[])
        quality = []
        for k, (tk, xk) in enumerate(zip(times, states)):
            R_ei = cfg.earth.rotation(tk).T
            sat_ecef = cfg.earth.rotation(tk) @ xk[:3]
            R_att = cfg.attitude.rotation(tk, xk)

            # --- candidate inlier landmarks: random pixels cast onto the ellipsoid
            n_try = cfg.inliers_per_frame
            uv_c = ol.random_pixel(rng, n_try, cam)
            d_ecef = (cam.unproject(uv_c) @ R_att) @ R_ei
            lm_true, hit = ray_ellipsoid(sat_ecef, d_ecef)
            hit &= sees_point(sat_ecef, lm_true)
            lm_true, uv_c = lm_true[hit], uv_c[hit]

            # --- camera quality at this frame (settings, illumination, blur)
            lat, lon, _ = ecef_to_geodetic(lm_true) if len(lm_true) else (np.empty(0),) * 3
            up_eci = surface_normal(lat, lon) @ cfg.earth.rotation(tk) if len(lm_true) else np.empty((0, 3))
            elev = np.arcsin(np.clip(up_eci @ sun.direction_eci(tk), -1, 1)) if len(lm_true) else np.empty(0)
            rng_m = np.linalg.norm(lm_true - sat_ecef, axis=1)
            q = cfg.quality.evaluate(cfg.settings_at(k), rng_m, np.linalg.norm(xk[3:]), np.linalg.norm(xk[:3]),
                                     elev, cam.focal_px, cfg.pixel_sigma)
            quality.append(q)

            detected = rng.random(len(lm_true)) < q.p_detect
            lm_true, uv_c, rng_m = lm_true[detected], uv_c[detected], rng_m[detected]
            sig_img = q.sigma_px[detected]
            n_in = len(lm_true)

            # --- inliers: noisy pixel of the true point; landmark reported with geolocation error
            sig_i = sig_img * np.exp(cfg.sigma_jitter * rng.standard_normal(n_in)) if cfg.sigma_jitter else sig_img
            uv_in = uv_c + sig_i[:, None] * rng.standard_normal((n_in, 2))
            lm_rep = lm_true + cfg.landmark_sigma_m * rng.standard_normal((n_in, 3))
            sig_rep = np.hypot(sig_i, landmark_sigma_to_pixels(cfg.landmark_sigma_m, rng_m, cam.focal_px))

            # --- outliers
            f = cfg.outlier_fraction
            f_eff = 1.0 - (1.0 - f) * (1.0 - (q.p_outlier_extra[0] if len(q.p_outlier_extra) else 0.0))
            n_out = int(round(n_in * f_eff / (1.0 - f_eff))) if f_eff < 1 and n_in else 0
            kinds = rng.choice(2, size=n_out, p=np.asarray(cfg.outlier_mix) / np.sum(cfg.outlier_mix))
            uv_o, lm_o_true, lm_o_rep, sig_o = np.empty((0, 2)), np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
            if n_out:
                # the underlying scene point of an outlier is a fresh random visible point
                uv_s = ol.random_pixel(rng, n_out * 3 + 5, cam)
                d_s = (cam.unproject(uv_s) @ R_att) @ R_ei
                lm_s, hit_s = ray_ellipsoid(sat_ecef, d_s)
                hit_s &= sees_point(sat_ecef, lm_s)
                lm_s, uv_s = lm_s[hit_s][:n_out], uv_s[hit_s][:n_out]
                n_out = len(lm_s)
                kinds = kinds[:n_out]
                uv_o = np.where((kinds == 0)[:, None], ol.random_pixel(rng, n_out, cam), uv_s)
                near = ol.displaced_landmark(rng, lm_s, cfg.wrong_landmark_offset_m)
                lm_o_rep = np.where((kinds == 1)[:, None], near, lm_s)
                lm_o_true = lm_s
                rng_o = np.linalg.norm(lm_s - sat_ecef, axis=1)
                sig_o = np.hypot(cfg.pixel_sigma, landmark_sigma_to_pixels(cfg.landmark_sigma_m, rng_o, cam.focal_px))
                sig_o = sig_o * cfg.outlier_sigma_inflate

            n_pts = n_in + n_out
            rows["t"].append(np.full(n_pts, tk))
            rows["uv"].append(np.vstack([uv_in, uv_o]))
            rows["lm_rep"].append(np.vstack([lm_rep, lm_o_rep]))
            rows["lm_true"].append(np.vstack([lm_true, lm_o_true]))
            rows["uv_true"].append(np.vstack([uv_c, np.full((n_out, 2), np.nan)]))
            rows["sig"].append(np.concatenate([sig_rep, sig_o]))
            rows["out"].append(np.concatenate([np.zeros(n_in, bool), np.ones(n_out, bool)]))
            rows["frame"].append(np.full(n_pts, k))

        cat = {k: np.concatenate(v) if k in ("t", "sig", "out", "frame") else np.vstack(v) for k, v in rows.items()}
        obs = Observations(cat["t"], cat["uv"], cat["lm_rep"], cat["sig"])
        return SimulatedData(obs, x0, times, states, cat["out"], cat["frame"].astype(int),
                             cat["lm_true"], cat["uv_true"], quality, cfg)
