"""Physical constants (SI units)."""

MU_EARTH = 3.986004418e14  # m^3/s^2
J2_EARTH = 1.08262668e-3
R_EARTH = 6378137.0  # m, equatorial
OMEGA_EARTH = 7.2921150e-5  # rad/s

# WGS84 ellipsoid
A_WGS84 = 6378137.0  # m, semi-major axis
F_WGS84 = 1.0 / 298.257223563
B_WGS84 = A_WGS84 * (1.0 - F_WGS84)
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84)

# Mean Earth-Sun geometry for the low-precision sun model
OBLIQUITY_RAD = 0.40909280422232897  # 23.4393 deg
YEAR_S = 365.25636 * 86400.0
