import numpy as np
import pytest

from astroflow.constants import R_EARTH
from astroflow.dynamics import kepler_to_cartesian

ORBITS = {
    "leo": (R_EARTH + 500e3, 0.001, np.radians(51.6), 0.7, 0.3, 1.0),
    "ecc": (R_EARTH + 3000e3, 0.2, np.radians(30.0), 1.0, 2.0, 0.5),
    "high_ecc": (26000e3, 0.7, np.radians(63.0), 0.2, 4.0, 2.0),
}


@pytest.fixture(params=list(ORBITS))
def x0(request):
    return kepler_to_cartesian(*ORBITS[request.param])
