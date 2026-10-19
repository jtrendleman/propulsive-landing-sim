# Changelog

## 1.8.0

The vehicle becomes a Falcon 9-class booster, modelled after the Falcon 9 Block 5 first stage on its single-engine
landing burn. Every number in config/ is now sourced, derived or a reasoned estimate, labelled S / D / E in its
comment and traced to docs/references.txt; none is a placeholder.

- Vehicle: 25.6 t dry, 8 t of landing propellant, one Merlin 1D-class engine (845 kN, 57-100 % throttle, Isp 282 s),
  3.66 m diameter, 47.7 m with the interstage, 18 m leg span, four grid fins, eight cold-gas thrusters. At its
  throttle floor the engine lifts 1.5-1.8 times the vehicle's weight, so it cannot hover: the landing is a
  single burn timed to reach the pad at touchdown speed.
- Two tanks (LOX above RP-1), each with a domed bottom and its own SP-106 slosh pendulum; propellant drains at the
  2.33 mixture ratio. Baffled tanks (damping 0.04). The state grows to 22 slots.
- Flight telemetry: data/telemetry/ holds four RTLS landings from SpaceX webcast telemetry (public domain);
  landingSim/analysis/telemetry.py finds the landing burn and fits the ballistic coefficient (2,040-2,170 kg/m^2),
  which sets the drag coefficient; scripts/compareTelemetry.py overlays the simulated landing on the real ones.
  The engine-off fall matches; the simulated burn starts lower and brakes harder (about 20 s against 30-33 s).
- Aerodynamics: the grid fins' drag acts at the fins (it is what keeps a base-first descent stable); body drag is
  shielded by the engine plume while the engine fires (central-nozzle retropropulsion, Korzun et al. 2008), as the
  telemetry's burn energy requires; crossflow drag and centres of pressure re-derived for the long body.
- Grid fins steer throughout, leading while their authority exceeds the gimbal's (dynamic pressure above the
  5.2 kPa crossover) and handing over below it.
- Contact: stick-slip friction (a foot pins an anchor; a stiff spring back to it, capped at mu N) replaces the
  regularised Coulomb law, which let a landed booster creep and, at this mass, limit-cycled numerically.
- Wind: the near-surface profile and its veer stop at the 1 km boundary layer; extrapolated to the 6 km start the
  veer had turned the wind 120 degrees.
- Sensors as flown on launch vehicles: navigation-grade IMU (HG9900 figures), RTK GPS against a base station at the
  landing zone, radar altimeter to 1.5 km, chamber pressure.
- Controller: hover-only logic (alignment hold, descent-rate law) stays for vehicles that can hover and switches off
  automatically for one that cannot, which flies the constant-deceleration stopping law near the ground. Attitude,
  position and lateral-guidance bandwidths are derived from the gimbal authority (2 rad/s attitude, 0.5 rad/s
  position, a 5 s floor on the lateral time-to-go).
- Criterion 3 is measured at the footprint centre, not the CG: on a 47 m booster a degree of rocking moves the CG
  tens of centimetres while the feet stay put.
- Windows support: UTF-8 file I/O throughout, a cross-platform detached launcher with --awake, and
  scripts/optimizeAll.py for the full tuning pipeline.
- Untuned baseline, 20 draws: KSC average 30 % succeeded (hard failures 10 %), KSC strong 10 %; most misses land
  0.5-1.5 m from the pad. The guidance and wind-schedule optimizers re-tune for this vehicle
  (scripts/optimizeAll.py); their validated results go here when the run completes.
- The PPO / SAC comparisons in checkpoints/ were flown on the 1.4 t vehicle and need retraining.

## 1.7.0

Wind-scheduled guidance, and two physics corrections that change every earlier number.

- Diagnosis first: flying the KSC strong Monte Carlo with a perfect wind forecast (true mean, profile and
  direction) gave exactly the baseline's 63 %, landing spots moving by centimetres. Wind speed was still the
  strongest predictor of failure (Cohen's d +1.68), so the flight software knew the wind well enough; its
  fixed knobs did nothing with it.
- Wind schedule (`[windSchedule]`, `landingSim/gnc/controller.py`): the controller converts its wind knowledge
  (forecast plus the observer's push correction divided by the push one m/s of crossflow makes at the current
  airspeed, held below 20 m/s) into a wind speed at the 10 m reference height, takes the part blowing away from
  the pad (fading to the full speed within 5 m of it, so the input stays continuous), and blends eight guidance
  knobs (`referenceDecel`, `ignitionMargin`, `lateralLead`, `maxTilt`, `highTilt`, `windFeedforwardGain`,
  `kPosition`, `kVelocity`) from their [guidance] values at 4.5 m/s to the strong-wind values at 7.5 m/s.
  Continuous, so there is no switching boundary to chatter on. `scripts/optimizeSchedule.py` tunes the
  strong-wind end and writes `config/earthBoosterScheduled.toml`.
- Physics correction 1, slosh: the 1.5.0 model applied the slosh spring force to a rigid body that already
  contained the slosh mass (hiding about a quarter of every lateral force, wind included, from both the vehicle
  and its IMU), used the 1 g frequency for the force but the real axial acceleration for the motion, and put the
  near-empty propellant in a 1 to 3 cm film across a flat 3.6 m floor, where the linear model swung it 11 to 31 m.
  Replaced by a domed tank (`landingSim/vehicle/tank.py`: half-ellipsoid bottom, depth 0.707 R, closed-form fill)
  and an SP-106 equivalent pendulum sized from the pool (400 kg: 330 kg on a 2.68 m pendulum, 1.91 rad/s at 1 g,
  against 1.96 for a lump sliding in the bowl), full sin / cos restoring force, rod tension acting on the tank at
  the hinge, a wall stop, and the rigid body as total minus slosh mass. Mass properties use the dome geometry.
- Physics correction 2, contact: the friction regularisation speed goes from 0.02 to 0.1 m/s. At 0.02 the yaw
  friction was too stiff for the 2.5 ms contact substep and limit-cycled; with the dome's smaller roll inertia a
  landed vehicle kept turning at 0.02 rad/s, just above the rest threshold, and timed out.
- Held-out results (100 draws): KSC average, optimized 97 / 99 % (seeds 1 / 2, 59 % used; 1.6.0 had 95 / 95 % at
  62 %), wind-scheduled 98 / 99 % at 58 %. KSC strong, optimized without the schedule 72 %, wind-scheduled
  96 / 90 % at 68 % used (hard failures 22 % -> 1 to 4 %). Seed 1 chose among the five confirmed candidates, so
  seed 2's 90 % is the unbiased number: close to the gate, not through it. KSC extreme 4 % -> 24 %. The tuned
  strong-wind knobs: tilt 21° low and 40° high (the search's upper bound), lateral lead 4.4, braking 2.9 m/s²,
  feedforward 0.97, kPosition 1.0, kVelocity 2.3.
- The PPO / SAC comparisons in `checkpoints/` (through V1.6.3) were flown on the 1.6.0 physics and need
  retraining before they say anything about 1.7.0.
- 208 tests.

## 1.6.0

Requirement: the flight software gets only what a real booster's sensors would give it.

- Sensor suite (`landingSim/gnc/sensors.py`, the only module that reads the true state): body-frame IMU with
  tactical-grade bias and noise (HG1700 order: gyro ~1°/h, accel ~1 mg), RTK GPS fixes that arrive 0.1 s late,
  a radar altimeter with ~1 % of range noise, a chamber-pressure transducer (thrust with 1 % calibration and 0.5 %
  noise), tank gauging (2 % bias, 1 kg noise), gimbal and throttle-valve position feedback, and a single attitude
  alignment at the start of the landing.
- Navigation integrates the gyros for attitude (bias now becomes slow attitude drift) and rotates the body-frame
  accelerometer with that estimate; delayed GPS fixes are carried forward with their velocity and the IMU
  acceleration (without the acceleration term the 0.1 s latency put a 1 m/s velocity error into a free fall).
- Wind: the controller no longer sees the configured mean wind. Each episode has a pre-flight forecast with
  1.5 m/s speed and 20° direction error, and a wind observer combines the push predicted from that forecast
  with a low-passed correction from what the IMU actually feels beyond measured thrust and still-air drag.
  Guidance leans against the estimate. The forecast prior matters most in the first seconds at 80 m/s, when
  the crosswind push is largest (about 4 m/s² in the strong case) and the observer has seen nothing yet.
- Thrust and mass are estimated separately: engine performance from chamber pressure against the measured
  throttle, vehicle mass from the IMU's axial specific force against measured thrust.
- The policy's observation adds the forecast wind, the wind-push estimate and the thrust estimate (21 values),
  all from the flight computer's side.
- `tests/testTruthBoundary.py` flies full landings with the true state wrapped in a proxy that raises on any
  read outside the sensor models, for both the classical controller and the policy-wrapped one.
- Search draws (40, seed 0): KSC average 95 % (68 % used); KSC strong 47.5 % without wind knowledge, 55 % with
  the observer alone, 62.5 % with the forecast prior, soft failures 17.5 % → 2.5 %. Hard failures (35 %) are
  unchanged: vehicles that drift tens of metres in the fast descent still run out of propellant flying back.
- Held-out results (seeds 1 / 2, 100 draws): KSC average 96 / 94 % at 2.0 m/s² (68 % used), 95 / 95 % at 3.0
  (62 %), 94 / 94 % at 3.5 (61 %). `earthBoosterOptimized.toml` moves to 3.0 m/s², the only setting that clears the
  gate on both seeds. KSC strong 63 % at 2.0 and 68 % at 3.0 (was 60 % in 1.5.0 with perfect wind knowledge of the
  mean); hard failures 30 %. KSC extreme 3 %. Nominal strong-wind landing now inside 0.5 m (0.31 m, was 0.67 m).
- Outputs: `results/` is gone. Every script writes to `outputs/VX.Y.Z-MM-DD-YY/` (simulator version from
  `landingSim.version`), and every training run report now includes a video of the policy's nominal flight.
- Overnight training (8 h each, KSC strong, residual limit 0.5 m/s²): PPO 23.2M steps, SAC 4.3M. On the same 100
  held-out draws the baseline succeeds 68 %, baseline + PPO 67 %, baseline + SAC 65 %. Hard failures stay at 25 to
  27 %: a bounded residual on the acceleration command cannot recover a vehicle that is short on propellant, so the
  remaining failures are an authority problem, not a learning one.
- 196 tests.

## 1.5.0

- Propellant slosh as the first-mode equivalent spring-mass of a cylindrical tank (NASA SP-106 / Abramson):
  slosh mass fraction, natural frequency (scaled by the current axial specific force) and mass station from the
  closed-form expressions in terms of fill depth and tank radius. Four new state variables (lateral displacement
  and rate of the slosh mass, body frame) widen the state array from 14 to 18; older logs load with a quiet tank.
  Damping ratio 0.01 is a placeholder for a smooth-wall tank without baffles. The spring and damper force on the
  tank wall acts at the mass station and is a producer like the others.
- Training infrastructure for long runs: `--hours`, periodic checkpoints under `steps/`, a held-out deterministic
  evaluation every `--eval-every` steps that keeps `best/model.zip`, and `evalHistory.png` in the run report.
- Held-out results (seeds 1 / 2, 100 draws): KSC average 95 / 94 % at 2.0 m/s² (65 % used), 96 / 94 % at 3.0
  (59 %), 93 / 93 % at 3.5 (57 %). Slosh costs about one point against 1.4.0. No setting above 2.0 confirms the gate
  on both seeds; `earthBoosterOptimized.toml` equals the baseline. The hard failures (4 to 5 % on every setting)
  are the windy tail of the average case and are the residual's training target.
- 187 tests. Overnight runs V1.5.0 (PPO, 8 workers) and V1.5.1 (SAC) launched on the strong-wind case.

## 1.4.0

- Contact: regularised Coulomb friction (μ 0.6, steel on concrete, fading below 2 cm/s so the vehicle can rest)
  replaces the purely viscous foot model; a progressive strut spring F = k x (1 + 4 (x/stroke)²); the pad is a
  tilted plane (Monte Carlo σ 0.6°, random downhill direction) and all contact is measured along its normal.
- Aero: drag coefficient rises with angle of attack, Cd(α) = Cd₀ (1 + 2 sin²α), and the centre of pressure moves
  1 m per radian toward the base. Placeholders in the shape DATCOM or Hoerner would give; no Mach dependence
  below Mach 0.25.
- Grid fins: 1.2 m² total at 8.5 m, side force q·A·CLα·δ with CLα 2 /rad and ±20° deflection (Washington &
  Miller order of magnitude). The controller allocates pitch and yaw torque to the fins first whenever dynamic
  pressure exceeds 100 Pa, then to the gimbal, then to the RCS, so the engine-off coast is now controllable.
- `Commands` gained `fins`; the command log is eight columns.
- What the new contact model exposed: with hardening 4 and 60 kN/m struts the load-factor pass rate fell to
  88 to 94 % on held-out draws (ordinary 2 to 3 m/s touchdowns peaking past 5 g). Retuned to hardening 1.0,
  50 kN/m, and a 0.10 m cutoff, which brought the search draws back to 100 % success and 100 % load-factor pass.
- Bug found by the remaining hard failures: the altimeter model computed the true base height with the
  controller's nominal CG, so a CG-perturbed vehicle read a biased altitude (16 cm in the traced case), the
  contact cutoff never fired, and the engine idled on the pad until the tank was dry. The sensors now ride on
  the real vehicle (`reseed(seed, trueSpacecraft=...)`); the controller still believes the nominal CG, which is
  the honest discrepancy. Present since 1.2.0.
- Aero finding: with the alpha-dependent model a crosswind hover sits at ~90° angle of attack, where two
  placeholders dominate. The centre of pressure at 90° was first set to mid-body (4.5 m), which put it 1.6 m
  above the CG and made the hovering vehicle weathervane against the controller (average case fell to 72 %).
  A booster's legs and engine section carry side area at the base, so the crossflow CP is set to 3.0 m. The
  near-ground tilt limit goes from 10° to 15° to give the crosswind hover authority to spare.
- Grid fins are used only during the engine-off coast: a fin deflection is also a sideways force at the top of
  the vehicle that guidance did not ask for, and with the engine lit the gimbal gives torque without it.
- Capability finding: with crossflow drag three times the axial value, the KSC extreme case (20 m/s aloft) drifts
  beyond the vehicle's lateral authority during the fast descent and is no longer flyable by this controller.
  The 1.3.0 figure of 64 % was flattered by the small-angle aero model. A third scenario,
  `earthBoosterKscStrong.toml` (7.5 m/s at 10 m, 14 m/s aloft), is added as the training target between the two.
- Held-out results (seeds 1 / 2, 100 draws): KSC average 96 / 95 % at 2.0 m/s² (64 % used), 96 / 94 % at 3.0
  (59 %), 92 / 94 % at 3.5 (57 %); the remaining hard failures are the windy tail (7.8 to 12 m/s at 10 m), where
  the hold runs to the propellant floor and sets down off the pad still sliding. KSC strong 68 % at 82 % used
  (30 % hard). KSC extreme 3 %. `earthBoosterOptimized.toml` returns to 2.0 m/s² for this version.
- 180 tests.

## 1.3.0

- Turbulence: a first-order Gauss-Markov gust process per axis (the Dryden spectrum's first-order form) with
  sigma = `turbulenceIntensity` times the local mean speed (0.15, MIL-HDBK-1797 low-altitude order of magnitude)
  and correlation time `turbulenceLength` / speed (200 m); vertical gusts at half the horizontal sigma.
- Wind shear: direction turns with altitude by `directionShearPerKm` (0.35 rad/km, Ekman-layer turning).
- The wind is now a per-episode `WindField` held by the drag producer and generated from the episode seed, so
  the Monte Carlo stays reproducible and an episode is deterministic inside RK4.
- Held-out results (seeds 1 / 2, 100 draws): KSC average 100 / 100 % at 2.0 m/s² (59 % used), 100 / 100 % at 3.0
  (53 %), 100 / 99 % at 3.5 (52 %). Gusts at this intensity cost nothing measurable against the 1.2.0 controller.
- 170 tests.

## 1.2.0

- Sensor models (`landingSim/gnc/sensors.py`): GPS position and velocity at 10 Hz, radar altimeter below 100 m,
  IMU specific force and gyro rates with per-episode constant biases, attitude solution noise. Tactical-grade
  IMU and RTK-class GPS orders of magnitude; values are placeholders until matched to datasheets.
- Navigation (`landingSim/gnc/navigation.py`): a per-axis Kalman filter fuses IMU acceleration with GPS fixes
  and the altimeter; the controller and the RL observation both fly the estimate. Every episode seeds its own
  sensor noise, so Monte Carlo draws stay reproducible and parallel runs match serial ones.
- Online thrust-to-mass estimate in the controller: the IMU's specific force along the body axis, minus modelled
  drag, against what the lag-modelled commanded throttle should produce, low-passed over 2 s. Commands are
  scaled by its inverse, which is the answer to the 1.1.0 weak-engine failures.
- `sensors.enabled = false` restores perfect navigation; the control-law unit tests use it.
- Finding: with 0.5 m (1σ) GPS the hard failures vanished (0 %) but soft failures rose to 12 to 13 % on both
  held-out seeds. A 0.5 m landing requirement cannot be met with 0.5 m position sensing; real pads use RTK or
  relative navigation. GPS noise set to the RTK level (3 cm, 5 cm/s), which is what the cited receiver delivers.
- Held-out results (seeds 1 / 2, 100 draws): KSC average 99 / 99 % at 2.0 m/s² (58 % used), 100 / 99 % at 3.0
  (53 %), 100 / 99 % at 3.5 (51 %). The thrust estimate made the harder burn robust: in 1.1.0 the same 3.5 m/s²
  confirmed only 90 / 85 %. `earthBoosterOptimized.toml` moves to 3.5 m/s².

## 1.1.0

- Actuator dynamics: ignition delay (0.3 s), first-order throttle lag (0.2 s), gimbal slew limit (the config's
  `maxGimbalRate`, 20°/s). Stepped once per logging step between the controller and the producers; the log now
  carries both commanded and achieved commands (`Trajectory.achieved`).
- International Standard Atmosphere (U.S. Standard Atmosphere 1976 troposphere) replaces the exponential model;
  thrust rises with altitude through `nozzleExitArea * (p0 - p)`, propellant flow stays throttle-proportional.
- Lateral CG offset (Monte Carlo σ 2 cm per axis) and thrust misalignment (σ 0.1° per axis) as constant torques
  the controller must trim; both enter the engine, gear and drag lever arms.
- What the ideal actuators were hiding: with the 1.0.1 gains the attitude loop limit-cycled against the 20°/s
  gimbal slew (commanded ±10° while the achieved angle lagged half a second) and tumbled without RCS; on the
  held-out set the baseline fell from 99 % to 79 % and propellant used rose from 60 % to 72 %.
- Retune for lag: attitude gains 20/12 → 10/8, ignition anticipates `ignitionDelay + throttleTimeConstant`,
  the alignment hold servos to a 2 m hold height instead of hovering wherever it is (a lean at 30 cm put a
  foot on the pad and forced a cutoff mid-correction), release thresholds 0.2 m and 0.3 m/s, lateral hold
  gains 1.5/2.4 → 1.0/2.0 (the stiffer hold hunted against the slew and burned 90 % of the propellant).
- Held-out results after the retune (seeds 1 / 2, 100 draws): KSC average 94 / 96 % at 2.0 m/s² (61 % used),
  93 / 91 % at 3.0 (52 %), 90 / 85 % at 3.5 (50 %). KSC extreme 67 % at 80 % used. Nominal average-day landing:
  1.9 m/s touchdown, 0.18 m offset, 40 % propellant left (was 41 %); nominal extreme-day landing leaves 11 %.
- Remaining failure mode: engines 3 to 14 % weak and vehicles up to 15 % heavy arrive 2.5 to 3.6 m/s fast with
  propellant to spare, because the controller assumes nominal thrust and the lag makes its corrections late.
  An online thrust estimate is the 1.2.0 answer. No braking setting above 2.0 confirms the gate in this version,
  so `earthBoosterOptimized.toml` equals the baseline for now.
- 158 tests.

Versions follow the training-run folders (`checkpoints/VX.Y.Z-MM-DD-YY`). Numbers quoted are from the
100-draw held-out confirmation (seed 1) on the KSC average case unless stated.

## 1.0.1

- Simulation speed: closed-form 3-vector cross product and quaternion rotation, vectorised landing gear
  with an out-of-reach early exit. Nominal mission 3.88 s to 1.04 s per landing (3.7x); no model change,
  trajectories match the previous version to 5e-14.
- Sensitivity analysis: `scripts/runSensitivity.py` reports which perturbation separates the failures and a
  one-at-a-time sweep at ±2σ.
- Fourth success criterion: peak sensed acceleration against `loadFactorLimit`. It exposed three placeholders:
  strut stiffness (200 kN/m peaked near 8 g at touchdown; now 60 kN/m), strut damping (5 kN·s/m added a 2.9 g
  jolt at first contact; now 2.5 kN·s/m), and the 0.5 m engine cutoff height, from which the vehicle free-fell
  to a 3.1 m/s touchdown (now 0.15 m, touchdown 1.9 m/s at 3.0 g). The peak sensed load is now the mission
  start, 3.5 g of drag plus thrust at 80 m/s.
- Held-out results (seeds 1 / 2, 100 draws each, four criteria): KSC average 99 / 99 % at 2.0 m/s² (60 % used),
  98 / 97 % at 3.0 (49 % used, the optimized setting), 97 / 93 % at 3.5. KSC extreme 82 % at 75 % used; the
  softer struts turned several slow off-pad touchdowns in 10 m/s surface wind into tip-overs (6 of 100), which
  the 1.0.0 stiff struts had absorbed. Contact realism is on the 1.4.0 list for that reason.
- 147 tests.

## 1.0.0

- Six-degree-of-freedom rigid body, RK4, force producers (engine, landing gear, drag, RCS), power-law wind
  profile with per-episode direction.
- ZEM/ZEV guidance with reference-profile time-to-go, lateral lead, altitude-scheduled tilt limit, coast and
  ignition for the hoverslam, terminal rate-of-descent and position hold with integrals, alignment hold.
- Success criteria, Monte Carlo with common random numbers, braking sweep, validated fuel optimizer.
- KSC average case: 98 % succeeded at 2.0 and 3.0 m/s² braking, 60 % and 48 % propellant used.
  KSC extreme case: 93 % at 72 % propellant used.
- Gymnasium environment and PPO/SAC/TD3 residual training with per-run report folders.
