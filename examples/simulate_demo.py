"""Build a scenario and write all plots to out/ (open the .html files in a browser)."""
from pathlib import Path

import numpy as np

from astroflow.constants import R_EARTH
from astroflow.sensor import CameraSettings, MatchQualityModel
from astroflow.sim import SimConfig, Simulator
from astroflow.viz import (animate_orbit, plot_ground_track, plot_orbit_3d, plot_pixel_tracks,
                           plot_settings_effect)

out = Path(__file__).resolve().parents[1] / "out"
out.mkdir(exist_ok=True)

cfg = SimConfig(
    orbit=(R_EARTH + 550e3, 0.002, np.radians(60.0), 0.3, 1.0, 0.5),
    n_frames=60, frame_dt=100.0,            # 100 minutes: about one orbit
    inliers_per_frame=25, outlier_fraction=0.2,
    pixel_sigma=0.5, landmark_sigma_m=30.0,
    quality=MatchQualityModel(), settings=CameraSettings(exposure_s=0.008, gain=1.5),
)
data = Simulator(cfg).generate()
print(f"{len(data.frame_times)} frames, {len(data.obs)} points: "
      f"{(~data.is_outlier).sum()} inliers, {data.is_outlier.sum()} outliers "
      f"({data.is_outlier.mean():.0%})")

for name, chart in {"orbit_3d": plot_orbit_3d(data), "orbit_animation": animate_orbit(data),
                    "ground_track": plot_ground_track(data), "pixel_tracks": plot_pixel_tracks(data),
                    "camera_settings": plot_settings_effect()}.items():
    chart.save(out / f"{name}.html")
    chart.save(out / f"{name}.png", scale_factor=2)
print("wrote", *sorted(p.name for p in out.iterdir()), sep="\n  ")
