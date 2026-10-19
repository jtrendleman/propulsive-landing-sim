""" testGuidance.py
ZEM/ZEV guidance, thrust-vector controller signs, and one closed-loop landing.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import pathlib

import numpy as np
import pytest

from landingSim.config import loadConfig
from landingSim.core import attitude
from landingSim.dynamics.simulation import Simulation
from landingSim.dynamics.state import State
from landingSim.dynamics.terminators import LandingTerminator
from landingSim.gnc.control import ThrustVectorController
from landingSim.gnc.controller import LandingController
from landingSim.gnc.guidance import ZemZevGuidance
from landingSim.vehicle.engine import Engine
from landingSim.vehicle.landingGear import LandingGear
from landingSim.vehicle.mass import massProperties

import dataclasses

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
eng = cfg.spacecraft.engine
# Control-law unit tests fly on the true state; sensor noise is covered by testNavigation.py.
perfect = dataclasses.replace(cfg, sensors=dataclasses.replace(cfg.sensors, enabled=False))
# The booster cannot hover (its throttle floor out-lifts it), so the hover-mode logic (rate-of-descent law, alignment
# hold, vertical trim) is checked on the same vehicle with a deep-throttling engine and the hover terminal settings.
hoverCapable = dataclasses.replace(
    perfect,
    spacecraft=dataclasses.replace(perfect.spacecraft, engine=dataclasses.replace(eng, minThrottle=0.2)),
    guidance=dataclasses.replace(perfect.guidance, terminalHeight=5.0, alignSpeed=0.3, alignOffset=0.2))


def state(position, velocity, quaternion=None):
    return State(np.array(position, float), np.array(velocity, float),
                 attitude.identity.copy() if quaternion is None else quaternion,
                 np.zeros(3), cfg.spacecraft.propellant.initialMass)


def testGuidanceFromRestAboveTargetCommandsLessThanOneG():
    aCmd = ZemZevGuidance(cfg.environment)(state([0, 0, 100.0], [0, 0, 0]), 0.0)
    assert 0.0 < aCmd[2] < cfg.environment.gravity
    assert np.allclose(aCmd[0:2], 0.0)


def testGuidanceFastDescentCommandsMoreThanOneG():
    aCmd = ZemZevGuidance(cfg.environment)(state([0, 0, 100.0], [0, 0, -60.0]), 0.0)
    assert aCmd[2] > cfg.environment.gravity


def testGuidanceSteersTowardTargetHorizontally():
    aCmd = ZemZevGuidance(cfg.environment)(state([50.0, 0, 200.0], [0, 0, -20.0]), 0.0)
    assert aCmd[0] < 0.0                                        # offset at +x, push toward -x


def testControllerZeroAccelerationIdlesEngine():
    commands = ThrustVectorController(cfg.spacecraft)(state([0, 0, 100.0], [0, 0, 0]), np.zeros(3), 0.0)
    assert commands.throttle == pytest.approx(eng.minThrottle)
    assert np.allclose(commands.gimbal, 0.0)


def testControllerUprightStraightUpNoGimbal():
    s = state([0, 0, 100.0], [0, 0, 0])
    aCmd = np.array([0.0, 0.0, 20.0])                          # above the throttle floor
    commands = ThrustVectorController(cfg.spacecraft)(s, aCmd, 0.0)
    expected = massProperties(s, cfg.spacecraft).total * 20.0 / (eng.maxThrust * eng.count)
    assert commands.throttle == pytest.approx(expected)
    assert np.allclose(commands.gimbal, 0.0, atol=1e-9)


def testControllerTiltedTowardPlusXGimbalsPositiveX():
    q = attitude.fromAxisAngle(np.array([0.0, 1.0, 0.0]), 0.1)  # body +z leans toward +x
    commands = ThrustVectorController(cfg.spacecraft)(state([0, 0, 100.0], [0, 0, 0], q), np.array([0.0, 0.0, 12.0]), 0.0)
    assert commands.gimbal[0] > 0.0                             # restores toward upright (step 5 sign)
    assert np.isclose(commands.gimbal[1], 0.0, atol=1e-9)


def testClosedLoopLandingFromTwoHundredMetres():
    gear = LandingGear(cfg.spacecraft)
    sim = Simulation(cfg, producers=[Engine(cfg.spacecraft), gear], dt=0.01, terminator=LandingTerminator(cfg.spacecraft, gear))
    cgHeight = massProperties(state([0, 0, 0], [0, 0, 0]), cfg.spacecraft).cgHeight
    traj = sim.run(state([0.0, 0.0, 200.0 + cgHeight], [0.0, 0.0, -20.0]), controller=LandingController(perfect))
    assert traj.endReason == "landed"
    final = traj.state(len(traj.time) - 1)
    assert np.linalg.norm(final.position[0:2]) < 0.5
    assert final.propellantMass > 0.05 * cfg.spacecraft.propellant.initialMass


def testClosedLoopMissionLanding():
    from landingSim.analysis.monteCarlo import runEpisode
    traj, result = runEpisode(perfect, LandingController(perfect))   # 6 km, 350 m/s down, 100 m off, every producer
    assert result.endReason == "landed"
    assert result.finalOffset < 1.0 and result.strutPass and result.loadFactorPass
    assert result.propellantFraction > 0.05


def testWithoutHoverItFliesTheStoppingLaw():
    controller = LandingController(perfect)
    cgHeight = massProperties(state([0, 0, 0], [0, 0, 0]), cfg.spacecraft).cgHeight
    falling = state([0.0, 0.0, cgHeight + 50.0], [0.0, 0.0, -30.0])      # feet 50 m up, 30 m/s down, inside the terminal phase
    assert not controller.canHover(controller.updateEstimate(falling, 0.0))
    commands = controller(falling, 0.0)
    vf = abs(perfect.guidance.descentRate)
    expectedAccel = cfg.environment.gravity + (30.0**2 - vf**2) / (2 * 50.0)
    total = massProperties(falling, cfg.spacecraft).total
    assert commands.throttle == pytest.approx(total * expectedAccel / (eng.maxThrust * eng.count), rel=1e-3)


def testTiltLimitIsScheduledWithAltitude():
    controller = LandingController(perfect)
    assert controller.tiltLimit(0.0) == pytest.approx(controller.maxTiltCommand)
    assert controller.tiltLimit(1000.0) == pytest.approx(controller.highTiltCommand)
    assert controller.maxTiltCommand < controller.tiltLimit(controller.tiltAltitude / 2) < controller.highTiltCommand


def testTerminalPhaseCommandsRateOfDescent():
    controller = LandingController(hoverCapable)
    cgHeight = massProperties(state([0, 0, 0], [0, 0, 0]), cfg.spacecraft).cgHeight
    hovering = state([0.0, 0.0, cgHeight + 2.0], [0.0, 0.0, 0.0])              # feet 2 m up, not moving
    commands = controller(hovering, 0.0)
    total = massProperties(hovering, cfg.spacecraft).total
    expectedAccel = cfg.environment.gravity + controller.kDescent * (controller.descentRate - 0.0)
    assert commands.throttle == pytest.approx(total * expectedAccel / (eng.maxThrust * eng.count), rel=1e-3)
    assert not controller.cutoff


def testTerminalPhaseHoldsPositionLaterally():
    controller = LandingController(hoverCapable)
    cgHeight = massProperties(state([0, 0, 0], [0, 0, 0]), cfg.spacecraft).cgHeight
    offset = state([2.0, 0.0, cgHeight + 2.0], [0.0, 0.0, -1.0])             # 2 m to +x, descending at the target rate
    commands = controller(offset, 0.0)
    assert commands.gimbal[0] > 0.0                                             # gx > 0 tips the nose toward -x, so thrust points at the pad


def hoverslamConfig(referenceDecel=18.0):
    import dataclasses
    return dataclasses.replace(perfect, guidance=dataclasses.replace(perfect.guidance, coast=True, referenceDecel=referenceDecel))


def testCoastKeepsEngineOffAboveBrakingAltitude():
    controller = LandingController(hoverslamConfig())
    high = state([0.0, 0.0, 6000.0], [0.0, 0.0, -300.0])           # braking altitude ~ 3 km, so coast
    commands = controller(high, 0.0)
    assert commands.throttle == 0.0 and np.allclose(commands.gimbal, 0.0)
    assert not controller.ignited
    g = perfect.guidance
    spool = eng.ignitionDelay + eng.throttleTimeConstant
    assert controller.brakingAltitude(high) == pytest.approx((300.0**2 - 1.0) / (2 * 18.0) * (1 + g.ignitionMargin) + g.terminalHeight
                                                             + 300.0 * spool + 0.5 * cfg.environment.gravity * spool**2)


def testBrakingAltitudeAnticipatesIgnitionDelay():
    import dataclasses
    slow = dataclasses.replace(hoverslamConfig(), spacecraft=dataclasses.replace(perfect.spacecraft, engine=dataclasses.replace(eng, ignitionDelay=eng.ignitionDelay + 1.0)))
    falling = state([0.0, 0.0, 5000.0], [0.0, 0.0, -300.0])
    assert LandingController(slow).brakingAltitude(falling) > LandingController(hoverslamConfig()).brakingAltitude(falling) + 250.0   # one more second at 300 m/s


def testIgnitionBelowBrakingAltitudeIsLatched():
    controller = LandingController(hoverslamConfig())
    low = state([0.0, 0.0, 2000.0], [0.0, 0.0, -300.0])
    assert controller(low, 0.0).throttle > 0.0 and controller.ignited
    high = state([0.0, 0.0, 6000.0], [0.0, 0.0, -300.0])
    assert controller(high, 1.0).throttle > 0.0                    # once lit, it stays lit


def testCoastHoldsAttitudeWithFinsThenRcs():
    q = attitude.fromAxisAngle(np.array([0.0, 1.0, 0.0]), 0.01)     # leaning toward +x while coasting
    fast = LandingController(hoverslamConfig())(state([0.0, 0.0, 6000.0], [0.0, 0.0, -300.0], q), 0.0)
    assert fast.throttle == 0.0
    assert fast.fins[0] < 0.0                                       # fast air: the fins tip the nose back toward -x
    slow = LandingController(hoverslamConfig())(state([0.0, 0.0, 6000.0], [0.0, 0.0, -30.0], q), 0.0)
    assert slow.throttle == 0.0 and np.allclose(slow.fins, 0.0)
    assert slow.rcs[1] < 0.0                                        # thin, slow air: the RCS does it


def testNoCoastIgnitesImmediately():
    import dataclasses
    noCoast = dataclasses.replace(perfect, guidance=dataclasses.replace(perfect.guidance, coast=False))
    controller = LandingController(noCoast)
    assert controller.ignited
    assert controller(state([0.0, 0.0, 1000.0], [0.0, 0.0, -30.0]), 0.0).throttle > 0.0


def testLaterHarderBurnLandsWithLessPropellant():
    from landingSim.analysis.monteCarlo import runEpisode
    gentleConfig, hardConfig = hoverslamConfig(16.0), hoverslamConfig(18.0)
    _, gentle = runEpisode(gentleConfig, LandingController(gentleConfig))
    _, slam = runEpisode(hardConfig, LandingController(hardConfig))
    assert gentle.endReason == "landed" and slam.endReason == "landed"
    assert slam.propellantFraction > gentle.propellantFraction


def testLateralChannelArrivesEarly():
    """With a lateral lead the sideways correction is stronger than with one shared time-to-go."""
    s = state([50.0, 0.0, 300.0], [0.0, 0.0, -25.0])
    shared = ZemZevGuidance(cfg.environment, referenceDecel=2.0, lateralLead=0.0)(s, 0.0)
    led = ZemZevGuidance(cfg.environment, referenceDecel=2.0, lateralLead=3.0)(s, 0.0)
    assert led[0] < shared[0] < 0.0
    assert led[2] == pytest.approx(shared[2])                        # vertical channel untouched


def testContactForcesCutoff():
    controller = LandingController(perfect)
    cgHeight = massProperties(state([0, 0, 0], [0, 0, 0]), cfg.spacecraft).cgHeight
    hardArrival = state([0.0, 0.0, cgHeight - 0.05], [0.0, 0.0, -5.0])   # feet 5 cm into the pad, still fast
    assert controller(hardArrival, 0.0).throttle == 0.0 and controller.cutoff


def testAlignmentHoldStopsDescentWhenOffPad():
    import dataclasses
    controller = LandingController(hoverCapable)
    cgHeight = massProperties(state([0, 0, 0], [0, 0, 0]), cfg.spacecraft).cgHeight
    offPad = state([1.0, 0.0, cgHeight + 2.0], [0.0, 0.0, -1.0])        # 1 m off, descending at the target rate
    onPad = state([0.0, 0.0, cgHeight + 2.0], [0.0, 0.0, -1.0])
    assert controller(offPad, 0.0).throttle > LandingController(hoverCapable)(onPad, 0.0).throttle   # hold = push harder than descend


def testAlignmentHoldGivesUpWhenPropellantIsLow():
    controller = LandingController(hoverCapable)
    cgHeight = massProperties(state([0, 0, 0], [0, 0, 0]), cfg.spacecraft).cgHeight
    nearlyDry = State(np.array([1.0, 0.0, cgHeight + 2.0]), np.array([0.0, 0.0, -1.0]), attitude.identity.copy(), np.zeros(3),
                      0.05 * cfg.spacecraft.propellant.initialMass)
    full = state([1.0, 0.0, cgHeight + 2.0], [0.0, 0.0, -1.0])
    assert controller(nearlyDry, 0.0).throttle < LandingController(hoverCapable)(full, 0.0).throttle


def testBaselineMeetsMissionSuccessRate():
    from landingSim.analysis.monteCarlo import runMonteCarlo, summarize
    optimized = loadConfig(repoRoot / "config" / "earthBoosterOptimized.toml")   # the tuned flight software, sensors on
    summary = summarize(runMonteCarlo(optimized, episodes=40, seed=0, workers=8))
    assert summary["missionSuccess"], summary


def testVerticalIntegralTrimsWeakEngine():
    """Holding at a sink rate below the commanded one, the integral raises the vertical command over time."""
    controller = LandingController(hoverCapable)
    cgHeight = massProperties(state([0, 0, 0], [0, 0, 0]), cfg.spacecraft).cgHeight
    sinking = state([0.0, 0.0, cgHeight + 3.0], [0.0, 0.0, -1.5])           # 0.5 m/s faster than the -1 m/s command
    first = controller(sinking, 0.0).throttle
    later = controller(sinking, 2.0).throttle
    assert later > first and controller.verticalIntegral > 0.0
