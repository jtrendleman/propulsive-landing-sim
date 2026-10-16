# Propulsive Landing Simulator

A 6-DOF propulsive landing simulator for a reusable booster on Earth. A classical
optimal guidance law (ZEM/ZEV) and an attitude controller fly the vehicle. A PPO
neural network learns residual corrections for wind, thrust error, mass error,
and CG offset.

## Success criteria

A landing passes only if all three hold:

1. Peak strut stress at touchdown, times the yield factor of safety, stays below
   the strut material yield strength.
2. At least 5% of the loaded propellant remains after the burn.
3. The vehicle settles within ±0.5 m of the target point.

## Setup

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
pytest
```

## Layout

| Folder | Contents |
|---|---|
| `config/` | Environment, spacecraft, and mission definitions (TOML). Every number starts here. |
| `landingSim/core/` | Quaternions, rotations, constants |
| `landingSim/environment/` | Gravity, atmosphere, wind |
| `landingSim/vehicle/` | Mass properties, engine, RCS, aerodynamics, landing gear |
| `landingSim/dynamics/` | 6-DOF equations of motion, RK4 integrator, simulation loop |
| `landingSim/gnc/` | Navigation, ZEM/ZEV guidance, attitude control |
| `landingSim/rl/` | Gymnasium environment, PPO training |
| `landingSim/analysis/` | Success criteria, Monte Carlo, plots |
| `scripts/` | Command-line entry points |
| `tests/` | Physics and unit tests |
| `docs/` | Physics notes and results |
