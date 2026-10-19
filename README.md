# Propulsive Landing Simulator

A 6-DOF propulsive landing simulator for a reusable booster on Earth. A classical
optimal guidance law (ZEM/ZEV) and an attitude controller fly the vehicle. A PPO
neural network learns residual corrections for wind, thrust error, mass error,
and CG offset.

## Success criteria

A landing passes only if all four hold:

1. Peak strut stress at touchdown, times the yield factor of safety, stays below
   the strut material yield strength.
2. At least 5% of the loaded propellant remains after the burn.
3. The vehicle settles within ±0.5 m of the target point.
4. Peak sensed acceleration stays below the structural load-factor limit.

## Setup

Python 3.12 (3.11 is the minimum: the config loader uses `tomllib`). ffmpeg is optional and only needed for the
`.mp4` videos; without it the animation scripts fall back to a GIF.

macOS / Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
pytest
winget install Gyan.FFmpeg            # optional, for the videos
```

The full test suite flies several hundred landings and takes about ten minutes.

## Layout

| Folder | Contents |
|---|---|
| `config/` | One TOML per scenario: environment, spacecraft, mission, Monte Carlo spreads, RL settings. Every number the simulation uses starts here. |
| `landingSim/core/` | Quaternions, rotations, constants |
| `landingSim/environment/` | Gravity, atmosphere, wind |
| `landingSim/vehicle/` | Mass properties, engine, RCS, aerodynamics, landing gear |
| `landingSim/dynamics/` | 6-DOF equations of motion, RK4 integrator, simulation loop |
| `landingSim/gnc/` | Navigation, ZEM/ZEV guidance, attitude control |
| `landingSim/rl/` | Gymnasium environment, PPO training |
| `landingSim/analysis/` | Success criteria, Monte Carlo, fuel optimizer, sensitivity, 3D figures |
| `landingSim/viewer/` | Interactive replay of a logged landing (self-contained HTML, three.js) |
| `scripts/` | Command-line entry points |
| `tests/` | Physics and unit tests |
| `docs/` | `references.txt`: every source and telemetry-derived value the configuration is built on |
| `data/telemetry/` | Falcon 9 webcast telemetry of four RTLS landings (public domain), for the comparison and the fitted values |

## Running

All scripts write their logs, plots and videos to `outputs/VX.Y.Z-MM-DD-YY/`, one folder per simulator version and day, the same convention as the training runs in `checkpoints/`.

```sh
python scripts/runLanding.py          # one closed-loop landing from the mission start, six-panel time history
python scripts/plotLanding3d.py       # 3D trajectory figure with vehicle glyphs, plus a four-window montage
python scripts/runMonteCarlo.py       # baseline controller over perturbed scenarios, pass rates and scatter
python scripts/plotDispersion3d.py    # trajectory bundle and touchdown-density surface (--run latest flies the newest policy)
python scripts/trainResidual.py       # residual policy, --algo ppo|sac|td3; each run gets checkpoints/VX.Y.Z-MM-DD-YY/ with model, log, settings
python scripts/evaluateResidual.py    # regenerate a run's report: Monte Carlo comparison, dispersion, nominal comparison, summary
python scripts/animateLanding3d.py    # the 3D trajectory figure as a video (landing3d.mp4)
python scripts/sweepBraking.py        # propellant used and success rate versus braking deceleration (gentle to hoverslam)
python scripts/optimizeGuidance.py    # least propellant with >= 95 % Monte Carlo success; --start referenceDecel=3.5 to begin at the sweep knee
python scripts/optimizeSchedule.py    # strong-wind end of the wind schedule, then held-out checks in average, strong and extreme wind
python scripts/optimizeAll.py         # both optimizers in sequence (several hours)
python scripts/compareTelemetry.py    # nominal landing against four real Falcon 9 RTLS landings from webcast telemetry
python scripts/runSensitivity.py      # which perturbation drives the failures: Monte Carlo separation and a +-2 sigma one-at-a-time sweep
python scripts/makeReplay.py          # interactive 3D replay page (landingReplay.html) from the newest landing log
python scripts/runDrop.py             # free fall; runHover.py and runDropOnGear.py are the other unit pictures
```

Long runs (optimization, training) can be started detached, so closing the terminal does not stop them, with
`--awake` keeping the computer from sleeping (caffeinate on macOS, the Windows power API on Windows):

```sh
python scripts/launchDetached.py --awake outputs/optimizeLog.txt -- python scripts/optimizeAll.py
python scripts/launchDetached.py --awake checkpoints/trainLog.txt -- python scripts/trainResidual.py --config config/earthBoosterKscStrong.toml --hours 7.5
```

It prints the process id: stop it with `kill <pid>` on macOS/Linux or `taskkill /PID <pid> /T /F` on Windows.
Use the virtual environment's Python (the one `pip install` went into) so the detached process finds the packages.

## How it works

- **Dynamics.** Rigid body with a quaternion attitude, integrated with RK4. Force producers
  (engine, landing gear, drag, RCS) share one interface: given the state and the commands they
  return a world-frame force, a body-frame torque, and a propellant flow. Gravity enters as an
  acceleration. Mass, CG, and inertia follow the propellant as the tank drains.
- **Propellant.** The tank has a domed (half-ellipsoid) bottom, so a nearly empty tank's propellant pools in the
  bowl the way it does in a real booster. Its first slosh mode is an SP-106 equivalent pendulum sized from that
  pool, swinging with the full gravity-like restoring force, pulling on the tank through its hinge, and stopped
  by the tank wall. The swinging mass is carried only through the pendulum, so it is never counted twice.
- **Flight software inputs.** Guidance, control and the learned policy see only what a real booster's sensors
  give: a body-frame IMU, RTK GPS fixes that arrive late, a radar altimeter, chamber pressure, tank gauging,
  actuator feedback, and a pre-flight wind forecast with error. `landingSim/gnc/sensors.py` is the only code
  that reads the true state, and `tests/testTruthBoundary.py` fails if anything else does. A Kalman filter
  builds the state estimate; separate estimators track engine performance, vehicle mass and the wind's push.
- **Guidance.** ZEM/ZEV (zero-effort miss / zero-effort velocity) gives the thrust acceleration
  that reaches the pad with the target velocity. Time-to-go comes from a constant-deceleration
  reference profile, which keeps it honest when drag has already slowed the vehicle.
- **Control.** The commanded acceleration sets the throttle and the desired body axis. A PD law
  on the attitude error becomes a torque, the engine gimbal delivers the pitch and yaw part through
  its lever arm, and the RCS takes roll and anything the gimbal clamp leaves over.
- **Hoverslam.** With `coast = true` the engine stays off until the braking altitude for the chosen
  deceleration, the RCS holds the vehicle upright through the coast, then ZEM/ZEV brakes along that
  profile. One number, `referenceDecel`, spans a gentle early burn to a late hard one.
- **Analysis.** Each logged episode is scored against the three criteria above. Monte Carlo
  perturbs thrust, dry mass, CG height, wind, and the initial offset; the controller keeps flying
  with the nominal numbers, so it never knows what changed. A configuration is a mission success
  only if at least 95 % of the draws succeed.
- **Fuel optimizer.** Minimises propellant used over the guidance knobs (braking deceleration,
  ignition margin, descent rate) with the 95 % rule as a penalty, every candidate flying the same
  Monte Carlo draws. The result is written out as a second scenario file.
- **Wind schedule.** In flight, the controller turns its wind knowledge (the forecast, corrected by the push the
  IMU feels) into a wind speed and takes the part blowing away from the pad. Eight guidance knobs blend
  continuously from their average-wind values at 4.5 m/s to strong-wind values at 7.5 m/s. A blend rather than
  a switch means there is no boundary for a noisy estimate to chatter across.
- **Residual learning.** A Gymnasium environment exposes the same randomized landing. The PPO
  action is an acceleration added to the guidance command every 0.1 s; the reward is the landing
  outcome with a small penalty for offset and effort. The trained policy drops into the same
  Monte Carlo loop as the baseline for a like-for-like comparison.

## Results so far

Version 1.8.0 replaced the vehicle with a Falcon 9-class booster (see `CHANGELOG.md`). The tables below are for
the 1.7.0 vehicle, a 1.4 t lander, and stay until the 1.8.0 tuning run validates the new numbers.

**Wind model.** Wind speeds are quoted at the 10 m reference height and grow with altitude by the power-law
profile v(z) = v₁₀·(z/10 m)^0.14, blowing in a direction drawn per episode. Two scenario files share the vehicle:

| scenario | 10 m mean speed | 1σ | at 1000 m | basis |
|---|---|---|---|---|
| `earthBooster.toml`, KSC average | 4.5 m/s | 2.0 m/s | 8.6 m/s | KSC tower climatology, 8.5 to 9.5 kt monthly means in the windy season |
| `earthBoosterKscStrong.toml`, KSC strong | 7.5 m/s | 1.0 m/s | 14 m/s | a windy day well inside the commit limit; the residual's training target |
| `earthBoosterKscExtreme.toml`, KSC extreme | 10.7 m/s | 1.0 m/s | 20 m/s | Falcon 9 launch commit limit, 30 mph sustained at 162 ft, brought to 10 m |

The 95 % requirement applies to the average case. The extreme case is a capability measurement.
Other Monte Carlo spreads (1σ): thrust 5 %, dry mass 5 %, CG height 0.2 m, initial offset 25 m.

**Nominal landings** (mission start, mean wind, no perturbations, version 1.7.0 with the wind schedule
`earthBoosterScheduled.toml`):

| case | touchdown speed | final offset | propellant left |
|---|---|---|---|
| KSC average | 1.4 m/s | 0.25 m | 45 % |
| KSC strong | 1.1 m/s | 0.35 m | 36 % |
| KSC extreme | tipped over 94 m downwind | | 8 % |

**Validated on 100 fresh draws per seed** (four criteria, version 1.7.0). The strong-wind knobs were tuned on
seed 0 and the best of five candidates chosen on seed 1, so seed 2 is the only set that played no part in tuning:

| case | flight software | succeeded (seed 1 / seed 2) | hard failed | propellant used | mission success |
|---|---|---|---|---|---|
| KSC average | optimized, 3.0 m/s² braking | 97 % / 99 % | 1 % | 59 % | yes |
| KSC average | **wind-scheduled** | **98 % / 99 %** | 0 to 1 % | **58 %** | **yes** |
| KSC strong | optimized, no schedule | 72 % | 22 % | 74 % | no |
| KSC strong | **wind-scheduled** | **96 % / 90 %** | 1 to 4 % | **68 %** | seed 1 only |
| KSC extreme | optimized, no schedule | 4 % | 90 % | 81 % | capability |
| KSC extreme | wind-scheduled | 24 % | 57 % | 83 % | capability |

In strong wind the schedule leans harder and earlier than the average-wind tuning: tilt limits of 21° near the
ground and 40° aloft (from 15° and 30°), a longer lateral lead (4.4 from 3.0) and about the same 2.9 m/s² braking.
It turns a third of the strong-wind draws from hard failures into landings and uses less propellant doing it,
without costing the average case anything. On the untouched seed it lands 90 %, so strong wind is close to the
95 % gate but not yet through it; the 40° high-altitude tilt sits on the edge of the search range.

Each version's numbers are in `CHANGELOG.md`. Version 1.7.0 corrected two physics errors that had made every
earlier number pessimistic: the 1.5.0 slosh model counted the sloshing propellant twice (hiding about a quarter
of every sideways force and letting a 2 cm film swing tens of metres), and the contact friction was too stiff
for its time step, which kept some vehicles that had already landed spinning slowly enough to time out.

**Method note.** Candidates are tuned on one Monte Carlo seed and confirmed on another. The in-sample
optimum always sits on the edge of the feasible region and loses a few points on fresh draws, so
`scripts/optimizeGuidance.py` confirms the cheapest in-sample candidates on held-out draws and only writes
one that still passes.

**Learned residual.** The PPO and SAC runs so far (through 1.6.0, before the physics corrections) equalled the
baseline but never beat it: an acceleration residual capped at 0.5 m/s² cannot recover a vehicle that is short of
propellant. Those comparisons need retraining against the 1.7.0 physics. `scripts/trainResidual.py` runs PPO, SAC or TD3 against it with wall-clock limits, checkpoints, a held-out
evaluation that keeps the best model, and a full report per run under `checkpoints/`.

Figures and video in `outputs/<version>/`: `landing.png` (time histories), `landing3d.png` and `landing3dMontage.png`
(3D trajectory with body-axis and thrust glyphs), `landing3d.mp4` (the same as a video),
`landingKscExtreme3d.png` (the extreme case), `monteCarlo.png` (outcome scatter), `dispersion3d.png`
(trajectory bundle and touchdown-density surface), `brakingSweep.png` (fuel against reliability), `sensitivity.png` (which perturbation drives the failures).
