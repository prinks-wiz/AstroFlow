"""What the camera sees: pixel tracks, and the modeled effect of exposure / gain / white balance."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from ._data import points_df
from ..sensor import CameraSettings, MatchQualityModel

KIND_COLORS = alt.Scale(domain=["inlier", "outlier"], range=["#2a9d8f", "#d1495b"])


def plot_pixel_tracks(data) -> alt.Chart:
    """All matched pixels in the image plane, colored by frame; the right panel marks outliers.
    Drag on the left panel to pick pixels and highlight them on the right."""
    cam = data.config.camera
    df = points_df(data)
    brush = alt.selection_interval(encodings=["x", "y"])
    x = alt.X("u:Q", scale=alt.Scale(domain=[0, cam.width]), title="u (px)")
    y = alt.Y("v:Q", scale=alt.Scale(domain=[cam.height, 0]), title="v (px)")  # image y points down
    frame = alt.Chart(pd.DataFrame(dict(u=[0, cam.width, cam.width, 0, 0], v=[0, 0, cam.height, cam.height, 0], o=range(5)))
                      ).mark_line(color="#888", strokeDash=[4, 3]).encode(x=x, y=y, order="o:Q")
    tip = ["frame", "kind", alt.Tooltip("u:Q", format=".1f"), alt.Tooltip("v:Q", format=".1f"),
           alt.Tooltip("sigma_px:Q", format=".2f")]
    left = alt.Chart(df).mark_circle(size=28, opacity=0.85).encode(
        x=x, y=y, color=alt.Color("frame:Q", scale=alt.Scale(scheme="viridis"), title="frame"),
        tooltip=tip).add_params(brush)
    right = alt.Chart(df).mark_point(filled=True, size=40).encode(
        x=x, y=y, color=alt.Color("kind:N", scale=KIND_COLORS, title="match"),
        shape=alt.Shape("kind:N", legend=None), tooltip=tip,
        opacity=alt.condition(brush, alt.value(0.95), alt.value(0.15)))
    w, h = 420, int(420 * cam.height / cam.width)
    return alt.hconcat((frame + left).properties(width=w, height=h, title="Pixel locations by frame"),
                       (frame + right).properties(width=w, height=h, title="Inliers vs outliers")
                       ).resolve_scale(color="independent", shape="independent")


def settings_effect_df(quality: MatchQualityModel, *, range_m=550e3, speed_mps=7600.0, radius_m=6928e3,
                       sun_elev_deg=60.0, focal_px=1662.0, base_sigma_px=0.5,
                       exposures=None, gains=(1.0, 2.0, 4.0, 8.0), wb_cases=None) -> pd.DataFrame:
    exposures = np.logspace(-4, -1, 60) if exposures is None else np.asarray(exposures)
    wb_cases = wb_cases or {"balanced (1, 1, 1)": (1.0, 1.0, 1.0), "warm cast (1.8, 1, 0.5)": (1.8, 1.0, 0.5)}
    rows = []
    for wb_name, wb in wb_cases.items():
        for g in gains:
            for e in exposures:
                o = quality.evaluate(CameraSettings(exposure_s=float(e), gain=g, wb_gains=wb), [range_m], speed_mps,
                                     radius_m, np.radians(sun_elev_deg), focal_px, base_sigma_px)
                rows.append(dict(exposure_ms=e * 1e3, gain=g, white_balance=wb_name, p_detect=float(o.p_detect[0]),
                                 sigma_px=float(o.sigma_px[0]), blur_px=float(o.blur_px[0]),
                                 p_outlier_extra=float(o.p_outlier_extra[0]),
                                 expected_good=float(o.p_detect[0] * (1 - o.p_outlier_extra[0]))))
    return pd.DataFrame(rows)


def plot_settings_effect(quality: MatchQualityModel | None = None, **scene) -> alt.Chart:
    """Modeled P(good match), pixel sigma and motion blur vs. exposure, one line per gain,
    one row per white-balance case. The coefficients are model assumptions, not measured
    Arducam data (see sensor.py)."""
    df = settings_effect_df(quality or MatchQualityModel(), **scene)
    xs = alt.X("exposure_ms:Q", scale=alt.Scale(type="log"), title="exposure (ms)")
    color = alt.Color("gain:O", scale=alt.Scale(scheme="viridis"), title="gain")

    def line(sub, field, title, scale):
        return alt.Chart(sub).mark_line(strokeWidth=2).encode(
            x=xs, y=alt.Y(f"{field}:Q", title=title, scale=scale), color=color,
            tooltip=["gain", "white_balance", alt.Tooltip("exposure_ms:Q", format=".2f"),
                     alt.Tooltip(f"{field}:Q", format=".3f")]).properties(width=230, height=170)

    rows = []
    for wb in df.white_balance.unique():
        sub = df[df.white_balance == wb]
        rows.append(alt.hconcat(
            line(sub, "expected_good", "P(detected and correct)", alt.Scale(domain=[0, 1])),
            line(sub, "sigma_px", "pixel sigma (px)", alt.Scale(type="log")),
            line(sub, "blur_px", "motion blur (px)", alt.Scale(type="log")),
        ).properties(title=f"White balance: {wb}"))
    return alt.vconcat(*rows).resolve_scale(color="shared")
