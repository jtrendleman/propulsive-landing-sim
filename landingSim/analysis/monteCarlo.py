""" monteCarlo.py
Perturbs the scenario config, flies one episode per draw, and summarizes the pass rate.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0

The controller is built from the NOMINAL config and the vehicle from the PERTURBED one, so the
controller flies a rocket that is slightly different from the one it believes it has. That gap
is what the RL residual is trained to close.
"""

import concurrent.futures
import dataclasses

import numpy as np

from landingSim.analysis.criteria import evaluate
from landingSim.dynamics.simulation import Simulation
from landingSim.dynamics.state import State, initialState
from landingSim.dynamics.terminators import LandingTerminator
from landingSim.environment.wind import WindField
from landingSim.gnc.controller import LandingController
from landingSim.vehicle.aero import Drag, GridFins
from landingSim.vehicle.engine import Engine
from landingSim.vehicle.landingGear import LandingGear
from landingSim.vehicle.mass import massProperties
from landingSim.vehicle.rcs import Rcs
from landingSim.vehicle.slosh import Slosh


def perturbConfig(config, rng):
    """One Monte Carlo draw: thrust, dry mass, CG height, wind speed (non-negative) and direction, and initial offset."""
    mc = config.monteCarlo
    spacecraft = config.spacecraft
    engine = dataclasses.replace(spacecraft.engine, maxThrust=spacecraft.engine.maxThrust * (1 + rng.normal(0, mc.thrustStd)),
                                 thrustMisalignment=tuple(float(v) for v in np.array(spacecraft.engine.thrustMisalignment) + rng.normal(0, mc.thrustMisalignmentStd, 2)))
    mass = dataclasses.replace(spacecraft.mass,
                               dryMass=spacecraft.mass.dryMass * (1 + rng.normal(0, mc.dryMassStd)),
                               cgHeightDry=spacecraft.mass.cgHeightDry + rng.normal(0, mc.cgOffsetStd),
                               cgLateralOffset=tuple(float(v) for v in np.array(spacecraft.mass.cgLateralOffset) + rng.normal(0, mc.cgLateralStd, 2)))
    windConfig = config.environment.wind
    direction = rng.uniform(0.0, 2 * np.pi) if mc.randomWindDirection else windConfig.direction
    wind = dataclasses.replace(windConfig, meanSpeed=max(rng.normal(windConfig.meanSpeed, windConfig.gustStd), 0.0), direction=direction)
    mission = dataclasses.replace(config.mission, initialHorizontalOffset=config.mission.initialHorizontalOffset + rng.normal(0, mc.initialOffsetStd))
    padSlope = abs(rng.normal(0.0, mc.padSlopeStd))
    padDirection = rng.uniform(0.0, 2 * np.pi)
    return dataclasses.replace(
        config,
        spacecraft=dataclasses.replace(spacecraft, engine=engine, mass=mass),
        environment=dataclasses.replace(config.environment, wind=wind, padSlope=padSlope, padSlopeDirection=padDirection),
        mission=mission,
    )


def missionStart(config):
    """Mission initial state with the CG lifted so initialAltitude is the feet altitude."""
    start = initialState(config.mission, config.spacecraft)
    cgHeight = massProperties(start, config.spacecraft).cgHeight
    return State(start.position + np.array([0.0, 0.0, cgHeight]), start.velocity, start.quaternion, start.omega, start.propellantMass)


def buildEpisode(config, dt=0.01, windSeed=None):
    """Fresh producers, terminator, and Simulation for one episode of this config. windSeed fixes the gust series."""
    gear = LandingGear(config.spacecraft, config.environment)
    windField = WindField(config.environment, windSeed, duration=config.mission.maxTime + 1.0)
    producers = [Engine(config.spacecraft, config.environment), gear, Drag(config.spacecraft, config.environment, windField),
                 GridFins(config.spacecraft, config.environment, windField), Rcs(config.spacecraft), Slosh(config.spacecraft)]
    sim = Simulation(config, producers, dt, terminator=LandingTerminator(config.spacecraft, gear))
    return sim, missionStart(config)


def runEpisode(config, controller, dt=0.01, sensorSeed=None):
    """sensorSeed seeds both the controller's sensors and the gust series (offset so the two streams differ)."""
    sim, start = buildEpisode(config, dt, None if sensorSeed is None else sensorSeed + 1)
    if hasattr(controller, "reseed"):
        controller.reseed(sensorSeed, trueSpacecraft=config.spacecraft, trueEnvironment=config.environment)   # sensors ride on the real vehicle
    trajectory = sim.run(start, controller=controller)
    return trajectory, evaluate(trajectory, config)


def _flyOne(job):
    perturbed, nominal, controllerFactory, dt, sensorSeed = job
    _, result = runEpisode(perturbed, controllerFactory(nominal), dt, sensorSeed)
    return result


def runMonteCarlo(config, controllerFactory=LandingController, episodes=None, seed=None, dt=0.01, progress=None, workers=1):
    """Returns a list of (perturbedConfig, LandingResult). controllerFactory(nominalConfig) builds a fresh controller.

    The draws depend only on the seed, so the same seed gives the same scenarios for any controller and
    any worker count (common random numbers). workers > 1 needs a picklable factory (a class or a
    module-level function, not a lambda).
    """
    episodes = config.monteCarlo.episodes if episodes is None else episodes
    seed = config.monteCarlo.seed if seed is None else seed
    rng = np.random.default_rng(seed)
    perturbedConfigs = [perturbConfig(config, rng) for _ in range(episodes)]
    jobs = [(perturbed, config, controllerFactory, dt, seed * 100_000 + i) for i, perturbed in enumerate(perturbedConfigs)]

    results = []
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(_flyOne, jobs)):
                results.append((perturbedConfigs[i], result))
                if progress is not None:
                    progress(i + 1, episodes, result)
    else:
        for i, job in enumerate(jobs):
            result = _flyOne(job)
            results.append((perturbedConfigs[i], result))
            if progress is not None:
                progress(i + 1, episodes, result)
    return results


missionSuccessRate = 0.95              # a configuration is a mission success only if this fraction of draws succeeds


def summarize(results):
    outcomes = [r.outcome for _, r in results]
    n = len(outcomes)
    landed = [r for _, r in results if r.endReason == "landed"]
    return {
        "episodes": n,
        "successRate": outcomes.count("succeeded") / n,
        "softFailRate": outcomes.count("softFailed") / n,
        "hardFailRate": outcomes.count("hardFailed") / n,
        "meanOffset": float(np.mean([r.finalOffset for r in landed])) if landed else float("nan"),
        "meanTouchdownSpeed": float(np.mean([r.touchdownSpeed for r in landed])) if landed else float("nan"),
        "meanPropellantFraction": float(np.mean([r.propellantFraction for _, r in results])),
        "strutPassRate": sum(r.strutPass for _, r in results) / n,
        "propellantPassRate": sum(r.propellantPass for _, r in results) / n,
        "offsetPassRate": sum(r.offsetPass for _, r in results) / n,
        "loadFactorPassRate": sum(r.loadFactorPass for _, r in results) / n,
        "meanPeakLoadFactor": float(np.mean([r.peakLoadFactor for _, r in results])),
        "meanPropellantUsed": float(np.mean([1.0 - r.propellantFraction for _, r in results])),
        "missionSuccess": outcomes.count("succeeded") / n >= missionSuccessRate,
    }
