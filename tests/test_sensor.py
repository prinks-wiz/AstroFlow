import numpy as np
import pytest

from astroflow.sensor import CameraSettings, MatchQualityModel

F = 1662.0
RANGE = np.array([550e3])
KW = dict(range_m=RANGE, speed_mps=7600.0, radius_m=6928e3, sun_elev_rad=np.radians(60), focal_px=F, base_sigma_px=0.4)


def ev(model=None, **settings):
    return (model or MatchQualityModel()).evaluate(CameraSettings(**settings), **KW)


def test_motion_blur_matches_hand_calculation():
    ground_speed = 7600.0 * 6378137.0 / 6928e3
    gsd = 550e3 / F
    out = ev(exposure_s=0.01)
    assert out.blur_px[0] == pytest.approx(ground_speed * 0.01 / gsd)
    assert ev(exposure_s=0.02).blur_px[0] == pytest.approx(2 * out.blur_px[0])


def test_exposure_tradeoff_signal_vs_blur():
    exposures = [0.0002, 0.001, 0.005, 0.02, 0.08]
    p = [ev(exposure_s=e).p_detect[0] for e in exposures]
    assert max(p) > p[0] and max(p) > p[-1]  # too short: no signal; too long: blur/saturation


def test_night_side_nothing_detected():
    out = MatchQualityModel().evaluate(CameraSettings(), **{**KW, "sun_elev_rad": np.radians(-20)})
    assert out.p_detect[0] == 0.0 and out.snr[0] == 0.0


def test_sigma_grows_with_blur():
    assert ev(exposure_s=0.03).sigma_px[0] > ev(exposure_s=0.003).sigma_px[0] > 0.4 - 1e-12


def test_high_gain_saturates():
    lo, hi = ev(gain=1.0), ev(gain=8.0)
    assert hi.saturated[0] > 1.0 >= lo.saturated[0]
    assert hi.p_detect[0] < lo.p_detect[0]


def test_white_balance_mismatch_adds_outliers_only_when_off():
    assert ev().p_outlier_extra[0] == 0.0
    m = ev(wb_gains=(1.6, 1.0, 0.6)).p_outlier_extra[0]
    assert 0 < m <= 1
    assert ev(wb_gains=(1.9, 1.0, 0.5)).p_outlier_extra[0] > m
    # a gain set equal to the reference balance (even if not 1,1,1) is not a mismatch
    model = MatchQualityModel(ref_wb_gains=(1.5, 1.0, 0.8))
    assert ev(model, wb_gains=(1.5, 1.0, 0.8)).p_outlier_extra[0] == 0.0


def test_disabled_model_is_passthrough():
    out = ev(MatchQualityModel.disabled(), exposure_s=0.5, gain=10.0, wb_gains=(3, 1, 0.2))
    assert out.sigma_px[0] == 0.4 and out.p_detect[0] == 1.0 and out.p_outlier_extra[0] == 0.0
