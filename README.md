# AstroFlow

Work out where a satellite is, and how it's moving, from photos taken by its own camera.

The camera spots known landmarks on the ground (a coastline, a lake, a city). Each match gives us a pixel in the image and a spot on Earth. From enough of these matches over time, we can solve for the satellite's orbit.

This repo is a research project comparing ways of doing that last step:

| Method | Idea | Status |
|---|---|---|
| Batch least squares | The classical approach: find the orbit that best explains all the matches. Robust loss functions (Huber, Cauchy) limit the damage from bad matches. | Planned |
| Particle filter | Track many candidate orbits at once and keep the ones that fit. | Planned |
| Flow-matching estimator (FMPE) | A learned model that outputs a whole distribution of plausible orbits, not just one answer. | Planned |

All three will sit behind the same interface and return the same kind of result (an orbit estimate with its uncertainty), so they can be compared fairly.

## Where things stand

| Step | What | Status |
|---|---|---|
| 1 | Orbit dynamics: how a satellite moves under gravity | **Done** |
| 2 | Camera, satellite pointing, Earth rotation, and the pixel prediction model | **Done** |
| 3 | Simulator: fake flights with adjustable noise, bad matches, and camera exposure / white balance | **Done** |
| 3b | Plots: 3D orbit, animation, ground track, pixel tracks | **Done** |
| 4 | Batch least squares and a rough starting guess | |
| 5 | First experiment: how accuracy changes with the number of matches and the share of bad ones | |
| 6 | Particle filter | |
| 7 | Flow-matching estimator | |

## What's built: orbit dynamics

Code lives in [src/astroflow/dynamics/](src/astroflow/dynamics/).

- **Force models** ([force_models.py](src/astroflow/dynamics/force_models.py)): how Earth pulls on the satellite. Point-mass gravity (`TwoBody`) and gravity including Earth's bulge (`J2Gravity`) are included. They can be combined, and drag or other forces can be added later.
- **Propagator** ([propagator.py](src/astroflow/dynamics/propagator.py)): moves a satellite's position and velocity forward or backward in time. It also returns the state transition matrix, which says how a small error in the starting state grows over time. The estimators need this to turn pixel errors into orbit corrections and to carry uncertainty forward.
- **Orbital elements** ([elements.py](src/astroflow/dynamics/elements.py)): converts between the usual orbit description (size, shape, tilt, ...) and position/velocity.

Units are meters, seconds and m/s, in an Earth-centered inertial frame.

### Example

```python
import numpy as np
from astroflow.constants import R_EARTH
from astroflow.dynamics import J2Gravity, Propagator, kepler_to_cartesian

# A 500 km orbit tilted 51.6 degrees: [x, y, z, vx, vy, vz]
x0 = kepler_to_cartesian(R_EARTH + 500e3, 0.001, np.radians(51.6), 0.5, 1.0, 0.0)

prop = Propagator(J2Gravity())
state, stm = prop.propagate(x0, t0=0.0, tf=3600.0)   # one hour later
```

`examples/propagate_demo.py` runs a longer example and prints how well energy is conserved.

### How we know it's right

The tests in [tests/dynamics/](tests/dynamics/) check the dynamics against independent references:

- The state transition matrix matches brute-force "nudge and re-run" numbers.
- Energy and angular momentum stay constant in the two-body case.
- Earth's bulge makes the orbit's plane slowly rotate. The measured rate matches the textbook formula to within 2%, including the sun-synchronous orbit case.
- Results agree with [Brahe](https://github.com/duncaneddy/brahe), an established astrodynamics library, to a few hundredths of a millimeter after three orbits.

## What's built: the simulator

It produces what a landmark-matching camera would hand to an estimator: for each frame, pixel
coordinates plus the matched landmark's position on Earth, with an uncertainty per point. The
truth is kept on the side so results can be scored.

```python
from astroflow.sim import SimConfig, Simulator

data = Simulator(SimConfig(inliers_per_frame=25, outlier_fraction=0.2, pixel_sigma=0.5, seed=1)).generate()
data.obs           # t, pixels, landmark_ecef, sigma_px  <- all an estimator gets to see
data.is_outlier    # which matches are wrong             <- for scoring only
```

Knobs: orbit, number and spacing of frames, inliers per frame, outlier fraction and type
(random pixel, or a landmark that is close but wrong), pixel noise, landmark position error,
per-point confidence, camera, pointing, and the seed (same seed gives identical data).

The pieces, all in [src/astroflow/](src/astroflow/):

- `frames.py`, `geodesy.py`: Earth rotation, WGS84 coordinates, horizon and ray-casting.
- `camera.py`, `attitude.py`: pinhole camera (default 1920x1080, 60 degree field of view),
  nadir-pointing or known-attitude pointing.
- `measurement.py`: predicts pixels and how they change with the orbit state. The same code
  makes the simulated data and will be the estimators' model, so the two cannot disagree.
- `sensor.py`: **camera settings** (exposure, gain, white balance) and their effect on match quality.
- `sim/`: the simulator and the outlier generators.

### Camera settings: read this before trusting the numbers

Exposure and white balance are modelled in two parts. Motion blur is plain geometry (how far the
ground slides during the exposure). The rest, such as how brightness, noise, saturation and a
white-balance mismatch change detection and outlier rates, is an **assumed model with adjustable
coefficients, not measured Arducam behaviour**. The camera itself is a generic placeholder until
the real unit's specs are known. A later stage will replace the assumed part with rendered images.

## Plots

```bash
pip install -e ".[viz]"
python examples/simulate_demo.py     # writes HTML and PNG files to out/ (no WebGL needed)
```

| File | Shows |
|---|---|
| `orbit_3d.html` | The orbit around the Earth; sliders rotate the view |
| `orbit_animation.html` | Same, plus a time slider: satellite moving, Earth turning, camera footprint |
| `ground_track.html` | Track on a world map, footprints, matched landmarks (inliers teal, outliers red) |
| `pixel_tracks.html` | Where matches land in the image; drag to highlight points |
| `camera_settings.html` | How exposure, gain and white balance change match quality (under the assumed model) |

## Getting started

You need Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,brahe,viz]"   # drop "brahe" / "viz" if you do not need them

pytest                           # run the tests
python examples/propagate_demo.py
```

## Layout

```
src/astroflow/
    constants.py        Earth constants
    dynamics/           orbit dynamics
    frames.py geodesy.py camera.py attitude.py measurement.py sensor.py illumination.py errors.py
    sim/                the simulator
    viz/                plots
tests/                  tests for everything above
examples/               small runnable demos
```

## License

See [LICENSE](LICENSE).
