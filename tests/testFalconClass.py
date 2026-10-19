""" testFalconClass.py
The Falcon 9-class booster's physics: telemetry extraction, plume drag shielding, the boundary-layer wind cap, and the fin/gimbal crossover.
Author: Jack Rendleman Date: 10/19/2026 Version: 0-1-0A
"""

import dataclasses
import pathlib

import numpy as np
import pytest

from landingSim.analysis.telemetry import isaDensity, landingBurn, loadTelemetry, missions
from landingSim.config import loadConfig
from landingSim.core import attitude
from landingSim.dynamics.state import Commands, State
from landingSim.environment.wind import windDirectionAt, windSpeedAt
from landingSim.vehicle.aero import Drag

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
telemetryFolder = repoRoot / "data" / "telemetry"


def testTelemetryLandingBurnsMatchTheRecordedNumbers():
    # docs/references.txt [D1], [D2]: ignition at 4.3-4.7 km and ~300-313 m/s, ~23-33 s to standstill, m / (Cd A) ~2,050-2,170 kg/m^2.
    for name in missions:
        burn = landingBurn(loadTelemetry(telemetryFolder, name))
        assert 4000.0 <= burn.ignitionAltitude <= 5000.0
        assert 290.0 <= burn.ignitionSpeed <= 320.0
        assert 20.0 <= burn.duration <= 35.0
        assert 1900.0 <= burn.ballisticCoefficient <= 2300.0


def testConfiguredDragMatchesTheTelemetryBallisticCoefficient():
    aero = cfg.spacecraft.aero
    massAtStart = cfg.spacecraft.mass.dryMass + cfg.spacecraft.propellant.initialMass
    beta = massAtStart / (aero.dragCoefficient * (aero.referenceArea + aero.gridFinArea))
    fitted = np.median([landingBurn(loadTelemetry(telemetryFolder, name)).ballisticCoefficient for name in missions])
    assert beta == pytest.approx(fitted, rel=0.1)


def testIsaDensityAtSeaLevelAndFiveKilometres():
    assert isaDensity(0.0) == pytest.approx(1.225, rel=1e-3)
    assert isaDensity(5000.0) == pytest.approx(0.736, rel=1e-2)


def testPlumeShieldsTheBodyFromDrag():
    falling = State(np.array([0.0, 0.0, 3000.0]), np.array([0.0, 0.0, -250.0]), attitude.identity.copy(), np.zeros(3), 8000.0)
    still = dataclasses.replace(cfg.environment, wind=dataclasses.replace(cfg.environment.wind, meanSpeed=0.0))
    drag = Drag(cfg.spacecraft, still)
    off = drag(falling, Commands(0.0, np.zeros(2)), 0.0)[0][2]
    on = drag(falling, Commands(0.8, np.zeros(2)), 0.0)[0][2]
    aero = cfg.spacecraft.aero
    finShare = aero.gridFinArea / (aero.referenceArea + aero.gridFinArea)
    assert on == pytest.approx(off * (finShare + (1 - finShare) * aero.plumeDragFactor), rel=1e-6)   # only the fins keep their drag


def testWindProfileAndVeerStopAtTheBoundaryLayer():
    wind = cfg.environment.wind
    top = wind.boundaryLayerHeight
    assert windSpeedAt(wind, 3 * top) == pytest.approx(windSpeedAt(wind, top))
    assert windDirectionAt(wind, 6000.0) == pytest.approx(wind.direction + wind.directionShearPerKm * top / 1000.0)
    assert windSpeedAt(wind, wind.referenceHeight) == pytest.approx(wind.meanSpeed)


def testFinGimbalCrossoverIsWhereTheirAuthoritiesMeet():
    sc = cfg.spacecraft
    from landingSim.vehicle.mass import massProperties
    from landingSim.analysis.monteCarlo import missionStart
    cg = massProperties(missionStart(cfg), sc).cgHeight
    finPerPascal = sc.aero.gridFinArea * sc.aero.finLiftSlope * sc.aero.maxFinDeflection * (sc.aero.finHeight - cg)
    gimbalAtFloor = sc.engine.minThrottle * sc.engine.maxThrust * np.sin(sc.engine.maxGimbalAngle) * cg
    assert sc.aero.finMinDynamicPressure == pytest.approx(gimbalAtFloor / finPerPascal, rel=0.1)
