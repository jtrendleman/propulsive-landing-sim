""" sensitivity.py
Which perturbation drives the failures: standardized separation per variable, and a one-at-a-time sweep at +-2 sigma.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

import dataclasses

import numpy as np
import matplotlib.pyplot as plt

from landingSim.analysis.monteCarlo import runEpisode
from landingSim.gnc.controller import LandingController


def perturbationTable(results):
    """Per-episode perturbation values from the perturbed configs, as a dict of arrays."""
    configs = [c for c, _ in results]
    nominalThrust = np.median([c.spacecraft.engine.maxThrust for c in configs])
    nominalMass = np.median([c.spacecraft.mass.dryMass for c in configs])
    nominalCg = np.median([c.spacecraft.mass.cgHeightDry for c in configs])
    offsetSign = np.array([np.sign(c.mission.initialHorizontalOffset) or 1.0 for c in configs])
    windX = np.array([c.environment.wind.meanSpeed * np.cos(c.environment.wind.direction) for c in configs])
    return {
        "wind speed (m/s)": np.array([c.environment.wind.meanSpeed for c in configs]),
        "wind toward pad (m/s)": -windX * offsetSign,                     # positive = blowing the vehicle from its offset toward the pad
        "thrust error (%)": 100 * (np.array([c.spacecraft.engine.maxThrust for c in configs]) / nominalThrust - 1),
        "dry mass error (%)": 100 * (np.array([c.spacecraft.mass.dryMass for c in configs]) / nominalMass - 1),
        "CG height error (m)": np.array([c.spacecraft.mass.cgHeightDry for c in configs]) - nominalCg,
        "initial offset (m)": np.array([abs(c.mission.initialHorizontalOffset) for c in configs]),
    }


def separation(results):
    """Standardized mean difference of each variable between failed and succeeded episodes (Cohen's d), sorted by |d|."""
    table = perturbationTable(results)
    failed = np.array([r.outcome != "succeeded" for _, r in results])
    rows = []
    for name, values in table.items():
        if failed.sum() == 0 or (~failed).sum() == 0:
            rows.append((name, 0.0)); continue
        pooled = np.sqrt(0.5 * (values[failed].var(ddof=1) + values[~failed].var(ddof=1))) if failed.sum() > 1 and (~failed).sum() > 1 else values.std()
        rows.append((name, float((values[failed].mean() - values[~failed].mean()) / pooled) if pooled > 0 else 0.0))
    return sorted(rows, key=lambda r: -abs(r[1]))


def oneAtATime(config, sigmas=2.0, controllerFactory=LandingController, dt=0.01):
    """Fly each perturbation alone at +-sigmas, everything else nominal; returns [(name, low result, high result)]."""
    mc, env, sc, mission = config.monteCarlo, config.environment, config.spacecraft, config.mission
    def wind(speed): return dataclasses.replace(config, environment=dataclasses.replace(env, wind=dataclasses.replace(env.wind, meanSpeed=max(speed, 0.0))))
    def windDir(direction): return dataclasses.replace(config, environment=dataclasses.replace(env, wind=dataclasses.replace(env.wind, direction=direction)))
    def thrust(f): return dataclasses.replace(config, spacecraft=dataclasses.replace(sc, engine=dataclasses.replace(sc.engine, maxThrust=sc.engine.maxThrust * f)))
    def mass(f): return dataclasses.replace(config, spacecraft=dataclasses.replace(sc, mass=dataclasses.replace(sc.mass, dryMass=sc.mass.dryMass * f)))
    def cg(d): return dataclasses.replace(config, spacecraft=dataclasses.replace(sc, mass=dataclasses.replace(sc.mass, cgHeightDry=sc.mass.cgHeightDry + d)))
    def offset(d): return dataclasses.replace(config, mission=dataclasses.replace(mission, initialHorizontalOffset=mission.initialHorizontalOffset + d))
    cases = [
        ("wind speed", wind(env.wind.meanSpeed - sigmas * env.wind.gustStd), wind(env.wind.meanSpeed + sigmas * env.wind.gustStd)),
        ("wind direction (with / against offset)", windDir(0.0), windDir(np.pi)),
        ("thrust", thrust(1 - sigmas * mc.thrustStd), thrust(1 + sigmas * mc.thrustStd)),
        ("dry mass", mass(1 - sigmas * mc.dryMassStd), mass(1 + sigmas * mc.dryMassStd)),
        ("CG height", cg(-sigmas * mc.cgOffsetStd), cg(+sigmas * mc.cgOffsetStd)),
        ("initial offset", offset(-sigmas * mc.initialOffsetStd), offset(+sigmas * mc.initialOffsetStd)),
    ]
    rows = []
    for name, low, high in cases:
        _, lowResult = runEpisode(low, controllerFactory(config), dt)
        _, highResult = runEpisode(high, controllerFactory(config), dt)
        rows.append((name, lowResult, highResult))
    return rows


def plotTornado(separationRows, oneAtATimeRows, path, title="Sensitivity"):
    fig, (axSep, axOat) = plt.subplots(1, 2, figsize=(13, 5.5))
    names = [r[0] for r in separationRows][::-1]
    values = [r[1] for r in separationRows][::-1]
    axSep.barh(names, values, color=["tab:red" if v > 0 else "tab:blue" for v in values])
    axSep.axvline(0, color="k", linewidth=1)
    axSep.set_xlabel("standardized difference, failed minus succeeded (Cohen's d)")
    axSep.set_title("What separates the failures (Monte Carlo)")
    axSep.grid(axis="x", alpha=0.4)
    labels = [r[0] for r in oneAtATimeRows][::-1]
    lowOffset = [min(r[1].finalOffset, 20.0) for r in oneAtATimeRows][::-1]
    highOffset = [min(r[2].finalOffset, 20.0) for r in oneAtATimeRows][::-1]
    y = np.arange(len(labels))
    axOat.barh(y - 0.2, lowOffset, height=0.4, color="tab:blue", label="-2 sigma / with offset")
    axOat.barh(y + 0.2, highOffset, height=0.4, color="tab:orange", label="+2 sigma / against offset")
    axOat.set_yticks(y); axOat.set_yticklabels(labels)
    axOat.axvline(0.5, color="k", linestyle="--", linewidth=1, label="0.5 m criterion")
    axOat.set_xlabel("final offset (m), capped at 20")
    axOat.set_title("One at a time, everything else nominal")
    axOat.legend(fontsize=8); axOat.grid(axis="x", alpha=0.4)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig
