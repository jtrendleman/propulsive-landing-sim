""" optimize.py
Fuel optimizer over the guidance knobs: least propellant used subject to a 95 % Monte Carlo success rate.
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.1.0

Every candidate flies the same Monte Carlo draws (same seed), so two candidates differ by design and
not by luck. The constraint enters as a penalty on the shortfall below the required success rate.
"""

import dataclasses
import pathlib

import numpy as np
from scipy.optimize import minimize

from landingSim.analysis.monteCarlo import missionSuccessRate, runMonteCarlo, summarize
from landingSim.gnc.controller import LandingController

penaltyWeight = 5.0                    # objective units (propellant fraction) per unit of success-rate shortfall


def withGuidance(config, **changes):
    return dataclasses.replace(config, guidance=dataclasses.replace(config.guidance, **changes))


def withSchedule(config, **changes):
    """Strong-wind knob values for the wind schedule, which is switched on."""
    return dataclasses.replace(config, windSchedule=dataclasses.replace(config.windSchedule, enabled=True, **changes))


tables = {"guidance": withGuidance, "windSchedule": withSchedule}


def evaluateCandidate(config, episodes, seed, workers=1, dt=0.01):
    return summarize(runMonteCarlo(config, LandingController, episodes=episodes, seed=seed, dt=dt, workers=workers))


def objective(summary):
    """Propellant used plus a penalty for falling short of the mission success rate."""
    shortfall = max(0.0, missionSuccessRate - summary["successRate"])
    return summary["meanPropellantUsed"] + penaltyWeight * shortfall


def sweepReferenceDecel(config, values, episodes, seed, workers=1, progress=None):
    """Returns [(referenceDecel, summary)] over the given braking decelerations."""
    rows = []
    for value in values:
        summary = evaluateCandidate(withGuidance(config, referenceDecel=float(value)), episodes, seed, workers)
        rows.append((float(value), summary))
        if progress is not None:
            progress(float(value), summary)
    return rows


def optimizeGuidance(config, bounds, episodes, seed, workers=1, maxEvaluations=40, progress=None, table="guidance"):
    """Nelder-Mead over the knobs in `bounds` ({name: (low, high)}) of config.<table>, starting from the config's values.

    Returns (bestConfig, bestSummary, history) with history = [(changes, summary, objective)].
    """
    names = list(bounds)
    low = np.array([bounds[k][0] for k in names], float)
    high = np.array([bounds[k][1] for k in names], float)
    x0 = np.array([getattr(getattr(config, table), k) for k in names], float)
    apply = tables[table]
    history = []

    def toChanges(u):
        values = low + np.clip(u, 0.0, 1.0) * (high - low)
        return {k: float(v) for k, v in zip(names, values)}

    def cost(u):
        changes = toChanges(u)
        summary = evaluateCandidate(apply(config, **changes), episodes, seed, workers)
        value = objective(summary)
        history.append((changes, summary, value))
        if progress is not None:
            progress(len(history), changes, summary, value)
        return value

    u0 = (x0 - low) / (high - low)
    initialSimplex = [u0] + [np.clip(u0 + 0.25 * np.eye(len(names))[i] * (1 if u0[i] < 0.5 else -1), 0, 1) for i in range(len(names))]
    minimize(cost, u0, method="Nelder-Mead", options={"maxfev": maxEvaluations, "xatol": 0.02, "fatol": 1e-3,
                                                       "initial_simplex": np.array(initialSimplex)})
    bestChanges, bestSummary, _ = min(history, key=lambda h: h[2])
    return apply(config, **bestChanges), bestSummary, history


def selectConfirmed(config, history, episodes, seed, workers=1, top=5, progress=None, table="guidance", bins=None):
    """Validated selection: confirm the `top` cheapest in-sample feasible candidates on fresh draws
    (a different seed) and return the cheapest that still meets the mission success rate.

    Returns (bestConfig, bestSummary, confirmations) with confirmations = [(changes, summary)]. If none
    confirms, the candidate with the highest confirmed success rate is returned instead. `bins` ({name: width})
    sets how coarsely candidates are told apart; the default suits the original guidance knobs.
    """
    apply = tables[table]
    feasible = [h for h in history if h[1]["successRate"] >= missionSuccessRate]
    # With nothing feasible in-sample, the cheapest candidate is just the one that gave up most often; rank by objective.
    ranked = sorted(feasible, key=lambda h: h[1]["meanPropellantUsed"]) if feasible else sorted(history, key=lambda h: h[2])
    # Nelder-Mead leaves many near-identical points around its final simplex; bin the knobs coarsely so
    # the confirmed set spans the feasible region instead of one cluster at the cliff edge.
    seen, candidates = set(), []
    for changes, _, _ in ranked:
        key = tuple(round(v / (bins[name] if bins else 0.1 if name == "referenceDecel" else 0.5), 0) for name, v in changes.items())
        if key not in seen:
            seen.add(key)
            candidates.append(changes)
        if len(candidates) == top:
            break

    confirmations = []
    for changes in candidates:
        summary = evaluateCandidate(apply(config, **changes), episodes, seed, workers)
        confirmations.append((changes, summary))
        if progress is not None:
            progress(changes, summary)

    passing = [c for c in confirmations if c[1]["missionSuccess"]]
    if passing:
        changes, summary = min(passing, key=lambda c: c[1]["meanPropellantUsed"])
    else:
        changes, summary = max(confirmations, key=lambda c: c[1]["successRate"])
    return apply(config, **changes), summary, confirmations


def _tomlValue(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_tomlValue(v) for v in value) + "]"
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def configToToml(config):
    """Serialise the nested config dataclasses back to TOML text (comments are not preserved)."""
    lines = []

    def emit(prefix, obj):
        scalars = [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj) if not dataclasses.is_dataclass(getattr(obj, f.name))]
        tables = [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj) if dataclasses.is_dataclass(getattr(obj, f.name))]
        if scalars:
            lines.append(f"[{prefix}]")
            lines.extend(f"{name} = {_tomlValue(value)}" for name, value in scalars)
            lines.append("")
        for name, value in tables:
            emit(f"{prefix}.{name}", value)

    for field in dataclasses.fields(config):
        emit(field.name, getattr(config, field.name))
    return "\n".join(lines)


def writeConfig(config, path, header=None):
    path = pathlib.Path(path)
    text = configToToml(config)
    if header:
        text = "".join(f"# {line}\n" for line in header.splitlines()) + "\n" + text
    path.write_text(text, encoding="utf-8")
    return path
