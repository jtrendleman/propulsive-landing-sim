""" controller.py
Closed-loop landing controller: coast, ignition, ZEM/ZEV guidance, optional residual, wind feedforward, limits, terminal descent, thrust-vector control, latched cutoff.
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.7.0

Works only from what the vehicle's sensors give it (see sensors.py): the navigation estimate, measured
chamber-pressure thrust, gimbal feedback, gauged propellant, and a pre-flight wind forecast with error.
Three estimators run on those measurements every step:
- engine performance: measured thrust over the thrust the measured throttle should give (nominal engine);
- vehicle mass: the IMU's axial specific force, less modelled still-air drag, against measured thrust;
- wind push: the push predicted from the pre-flight forecast (drag with the forecast wind minus drag in still
  air) plus a low-passed correction for what the IMU actually feels beyond thrust and still-air drag. The
  forecast prior matters most in the first seconds at 80 m/s, when the push is largest and the observer
  has seen nothing yet. Guidance leans against the total (windFeedforwardGain).

Wind schedule (config.windSchedule, if enabled): the flight software turns its wind knowledge into a wind
speed at the reference height (forecast, plus the observed push correction divided by how much push one
m/s of crossflow makes at the current airspeed), takes the part blowing away from the pad, and blends
the guidance knobs listed in windScheduleKnobs between their [guidance] values (average wind) and the
[windSchedule] values (strong wind). The blend is continuous, so there is no switching boundary to chatter on.

Every tuning number comes from config.guidance. Has memory (ignition and cutoff latches), so build a
new instance for every episode. `residual` is a world-frame acceleration added to the guidance
command; the RL policy sets it, everything else leaves it at zero.

Phases, top to bottom of __call__:
- Cutoff (latched): feet at the pad and descending slowly -> engine off for good.
- Coast (if guidance.coast): engine off, RCS holds the vehicle upright, until the feet altitude drops
  to the braking altitude (u^2 - vf^2) / (2 a) scaled by (1 + ignitionMargin), plus terminalHeight.
  This is the hoverslam: the braking deceleration a = referenceDecel sets how late ignition happens.
- Braking: ZEM/ZEV with a = referenceDecel as its reference profile.
- Terminal (below terminalHeight): PID position hold laterally; vertically a rate-of-descent law if the
  vehicle can hover (minimum thrust below its weight), otherwise the constant-deceleration stopping law
  a = g + (v^2 - vf^2) / (2 h), which reaches the touchdown speed vf exactly at the pad while asking for the
  same deceleration all the way down: a booster whose throttle floor out-lifts its weight cannot hold a
  descent rate, and on its stopping track the throttle stays where the burn put it. The
  integral starts accumulating below integralHeight so it has time to find the wind bias before the
  terminal phase; a PD hold alone always leaves offset = disturbance / kPosition in a steady wind.
- Limits: vertical floor, altitude-scheduled tilt limit.
"""

import dataclasses

import numpy as np

from landingSim.core import attitude
from landingSim.dynamics.state import Commands
from landingSim.gnc.control import ThrustVectorController
from landingSim.gnc.guidance import ZemZevGuidance
from landingSim.environment.atmosphere import airDensity
from landingSim.environment.wind import windSpeedAt, windVelocity
from landingSim.gnc.navigation import Navigator
from landingSim.vehicle.engine import Engine
from landingSim.vehicle.mass import massProperties

windScheduleKnobs = ("referenceDecel", "ignitionMargin", "lateralLead", "maxTilt", "highTilt", "windFeedforwardGain", "kPosition", "kVelocity")


def blendFactor(windSpeed, schedule):
    """0 at the average wind speed, 1 at the strong one, linear in between and clamped outside."""
    span = schedule.strongWindSpeed - schedule.averageWindSpeed
    if span <= 0.0:
        return 0.0
    return float(np.clip((windSpeed - schedule.averageWindSpeed) / span, 0.0, 1.0))


def blendedGuidance(guidanceConfig, schedule, s):
    """The [guidance] table with each scheduled knob moved fraction s of the way to its strong-wind value."""
    return dataclasses.replace(guidanceConfig, **{k: (1 - s) * getattr(guidanceConfig, k) + s * getattr(schedule, k)
                                                   for k in windScheduleKnobs})


class LandingController:
    def __init__(self, config, seed=None):
        self.config = config
        g = config.guidance
        self.navigator = Navigator(config, seed)
        self.estimate = None
        self.estimateTime = None
        self.forecastEnvironment = config.environment       # replaced per episode by a forecast with error
        self.engineScale = 1.0             # measured thrust / nominal thrust at the measured throttle
        self.massScale = 1.0               # nominal mass / estimated true mass
        self.thrustScale = 1.0             # engineScale * massScale: real acceleration per commanded acceleration
        self.windAccel = np.zeros(2)       # m/s^2, estimated sideways push of the wind (forecast prior + observed correction)
        self.windCorrection = np.zeros(2)  # m/s^2, low-passed observed push minus the forecast push
        self.baseGuidance = g
        self.guidanceConfig = g            # the scheduled knobs, re-blended every step when the wind schedule is on
        self.windCorrectionVelocity = np.zeros(2)   # m/s at the reference height, observed wind minus forecast wind
        self.windEstimate = np.zeros(2)    # m/s at the reference height, the flight software's best wind
        self.blend = 0.0                   # wind-schedule blend factor, 0 = average-wind knobs, 1 = strong-wind knobs
        self.guidance = ZemZevGuidance(config.environment, targetVelocity=(0.0, 0.0, g.descentRate),
                                       referenceDecel=g.referenceDecel, tgoMin=g.tgoMin, tgoMax=g.tgoMax, lateralLead=g.lateralLead,
                                       lateralTgoMin=g.lateralTgoMin)
        self.tvc = ThrustVectorController(config.spacecraft, g.kp, g.kd)
        self.nominalEngine = Engine(config.spacecraft, config.environment)
        self.maxTiltCommand = g.maxTilt
        self.highTiltCommand = g.highTilt
        self.tiltAltitude = g.tiltAltitude
        self.terminalHeight = g.terminalHeight
        self.descentRate = g.descentRate
        self.kDescent = g.kDescent
        self.ignited = not g.coast
        self.cutoff = False
        self.residual = np.zeros(3)
        self.lateralIntegral = np.zeros(2)
        self.verticalIntegral = 0.0
        self.lastTime = None
        self.lastThrottle = 0.0

    def guidanceCommand(self, state, t):
        """Pure ZEM/ZEV acceleration toward the pad, before the residual and the limits."""
        cgHeight = massProperties(state, self.config.spacecraft).cgHeight
        return self.guidance(state, t, targetPosition=np.array([0.0, 0.0, cgHeight]))

    def dynamicPressure(self, nav, t):
        """What the controller believes q is: ISA density and its own velocity estimate against the forecast wind."""
        vRel = nav.velocity - windVelocity(self.forecastEnvironment, nav.position, t)
        return 0.5 * airDensity(self.config.environment, nav.position[2]) * float(vRel @ vRel)

    def canHover(self, nav):
        """True if the engine at its throttle floor (as the thrust estimate believes it) lifts less than the weight."""
        eng = self.config.spacecraft.engine
        floor = eng.minThrottle * self.nominalEngine.maxThrustAt(nav.position[2]) * eng.count * self.engineScale
        weight = massProperties(nav, self.config.spacecraft).total / self.massScale * self.config.environment.gravity
        return floor < weight

    def tiltLimit(self, feetHeight):
        fraction = np.clip(feetHeight / self.tiltAltitude, 0.0, 1.0)
        return self.maxTiltCommand + fraction * (self.highTiltCommand - self.maxTiltCommand)

    def brakingAltitude(self, state):
        """Feet altitude at which braking at referenceDecel reaches the terminal speed at terminalHeight,
        plus the distance fallen while the engine lights and spools up (ignitionDelay + throttleTimeConstant)."""
        g = self.guidanceConfig
        eng = self.config.spacecraft.engine
        u = max(-state.velocity[2], 0.0)
        vf = abs(g.descentRate)
        spoolTime = eng.ignitionDelay + eng.throttleTimeConstant
        return (max(u**2 - vf**2, 0.0) / (2 * g.referenceDecel) * (1 + g.ignitionMargin) + g.terminalHeight
                + u * spoolTime + 0.5 * self.config.environment.gravity * spoolTime**2)

    def reseed(self, seed=None, trueSpacecraft=None, trueEnvironment=None):
        """New sensor noise for an episode, the vehicle the sensors ride on, and a wind forecast drawn around the true mean wind."""
        self.navigator.reseed(seed, trueSpacecraft, trueEnvironment)
        sc = self.config.sensors
        truth = (trueEnvironment if trueEnvironment is not None else self.config.environment).wind
        rng = np.random.default_rng(None if seed is None else seed + 7_919)
        forecastWind = dataclasses.replace(self.config.environment.wind,
                                           meanSpeed=max(truth.meanSpeed + rng.normal(0.0, sc.forecastSpeedError), 0.0),
                                           direction=truth.direction + rng.normal(0.0, sc.forecastDirectionError))
        self.forecastEnvironment = dataclasses.replace(self.config.environment, wind=forecastWind)

    def observeActuators(self, achieved):
        """The simulator reports what the actuators did last step; the sensors read it as chamber pressure and gimbal feedback."""
        self.navigator.observeActuators(achieved)

    def updateEstimate(self, state, t):
        """Run the sensors, navigation and the thrust/mass/wind estimators once per time step (cached for repeated calls)."""
        if self.estimateTime is not None and t == self.estimateTime:
            return self.estimate
        dt = 0.0 if self.estimateTime is None else max(t - self.estimateTime, 0.0)
        nav = self.navigator(state, t)
        self.estimate, self.estimateTime = nav, t
        self.updateEstimators(nav, dt)
        if self.config.windSchedule.enabled:
            self.updateWindSchedule(nav, massProperties(nav, self.config.spacecraft))
        return nav

    def updateEstimators(self, nav, dt):
        sc = self.config.sensors
        m = self.navigator.lastMeasurement
        if m is None or dt <= 0.0 or sc.thrustEstimateTimeConstant <= 0.0:
            return
        eng = self.config.spacecraft.engine
        props = massProperties(nav, self.config.spacecraft)
        zBody = attitude.rotateBodyToWorld(nav.quaternion, np.array([0.0, 0.0, 1.0]))
        specificWorld = attitude.rotateBodyToWorld(nav.quaternion, m.specificForceBody)
        aero = self.config.spacecraft.aero
        speed = np.linalg.norm(nav.velocity)
        stillDragAccel = (-0.5 * airDensity(self.config.environment, nav.position[2]) * aero.dragCoefficient
                          * (aero.referenceArea + aero.gridFinArea) * speed * nav.velocity) / props.total

        thrustAccel = np.zeros(3)
        if m.throttleFeedback >= eng.minThrottle and m.thrust > 0.0:
            nominalThrust = m.throttleFeedback * self.nominalEngine.maxThrustAt(nav.position[2])
            ratio = float(np.clip(m.thrust / nominalThrust, 0.7, 1.3))
            self.engineScale += (ratio - self.engineScale) * min(dt / sc.engineEstimateTimeConstant, 1.0)
            directionBody = attitude.normalize(np.array([np.tan(m.gimbalFeedback[0]), np.tan(m.gimbalFeedback[1]), 1.0]))
            thrustWorld = m.thrust * attitude.rotateBodyToWorld(nav.quaternion, directionBody)
            if m.throttleFeedback > 0.5:
                expectedAxial = float(thrustWorld @ zBody) / props.total
                measuredAxial = float((specificWorld - stillDragAccel) @ zBody)
                massRatio = float(np.clip(measuredAxial / expectedAxial, 0.8, 1.25))
                self.massScale += (massRatio - self.massScale) * min(dt / sc.thrustEstimateTimeConstant, 1.0)
            thrustAccel = thrustWorld / props.total * self.massScale
        self.thrustScale = self.engineScale * self.massScale
        forecastPush = self.forecastWindPush(nav, props)
        observedPush = (specificWorld - thrustAccel - stillDragAccel * self.massScale)[0:2]
        self.windCorrection += (observedPush - forecastPush - self.windCorrection) * min(dt / sc.windObserverTimeConstant, 1.0)
        self.windAccel = forecastPush + self.windCorrection

    def updateWindSchedule(self, nav, props):
        """Wind speed estimate at the reference height, its away-from-pad part, and the re-blended guidance knobs."""
        ws = self.config.windSchedule
        windConfig = self.config.environment.wind
        profile = windSpeedAt(dataclasses.replace(windConfig, meanSpeed=1.0), nav.position[2])   # speed at altitude per m/s at reference height
        forecast = windVelocity(self.forecastEnvironment, nav.position, 0.0)
        vRel = nav.velocity - forecast
        airspeed = np.linalg.norm(vRel)
        if airspeed > ws.minAirspeed:
            # One m/s of crossflow changes the drag force by 0.5 rho Cd(alpha) A |vRel| (the linearised drag law),
            # so the observed push correction divided by that is the wind correction. Below minAirspeed it holds.
            aero = self.config.spacecraft.aero
            zBody = attitude.rotateBodyToWorld(nav.quaternion, np.array([0.0, 0.0, 1.0]))
            sinAlpha2 = 1.0 - min(abs(zBody @ vRel) / airspeed, 1.0) ** 2
            pushPerWind = (0.5 * airDensity(self.config.environment, nav.position[2]) * aero.dragCoefficient
                           * (1 + aero.dragAlphaFactor * sinAlpha2) * (aero.referenceArea + aero.gridFinArea) * airspeed
                           / props.total * self.massScale)
            self.windCorrectionVelocity = self.windCorrection / pushPerWind / profile
        self.windEstimate = forecast[0:2] / profile + self.windCorrectionVelocity
        # The part blowing away from the pad, fading to the full speed inside padRadius where "away" has no direction
        # (a fade rather than a cut keeps the blend input continuous, so the knobs cannot jump at the radius).
        offset = nav.position[0:2]
        distance = np.linalg.norm(offset)
        fullSpeed = float(np.linalg.norm(self.windEstimate))
        away = float(self.windEstimate @ offset / distance) if distance > 1e-6 else fullSpeed
        weight = min(distance / ws.padRadius, 1.0) if ws.padRadius > 0.0 else 1.0
        windSpeed = weight * away + (1 - weight) * fullSpeed
        self.blend = blendFactor(windSpeed, ws)
        g = blendedGuidance(self.baseGuidance, ws, self.blend)
        self.guidanceConfig = g
        self.guidance.referenceDecel = g.referenceDecel
        self.guidance.lateralLead = g.lateralLead
        self.maxTiltCommand = g.maxTilt
        self.highTiltCommand = g.highTilt

    def forecastWindPush(self, nav, props):
        """Sideways acceleration the forecast wind should produce: drag with the forecast wind minus drag in still air."""
        aero = self.config.spacecraft.aero
        rho = airDensity(self.config.environment, nav.position[2])
        area = aero.referenceArea + aero.gridFinArea
        zBody = attitude.rotateBodyToWorld(nav.quaternion, np.array([0.0, 0.0, 1.0]))

        def drag(vRel):
            speed = np.linalg.norm(vRel)
            if speed < 1e-6:
                return np.zeros(3)
            sinAlpha2 = 1.0 - min(abs(zBody @ vRel) / speed, 1.0) ** 2
            return -0.5 * rho * aero.dragCoefficient * (1 + aero.dragAlphaFactor * sinAlpha2) * area * speed * vRel

        wind = windVelocity(self.forecastEnvironment, nav.position, 0.0)
        return ((drag(nav.velocity - wind) - drag(nav.velocity)) / props.total * self.massScale)[0:2]

    def __call__(self, state, t):
        g = self.guidanceConfig
        nav = self.updateEstimate(state, t)
        cgHeight = massProperties(nav, self.config.spacecraft).cgHeight
        feetHeight = nav.position[2] - cgHeight

        # 1. Engine cutoff, latched: slow and at the pad, or any ground contact (engine on in contact tumbles the vehicle).
        if self.cutoff or feetHeight <= 0.0 or (feetHeight <= g.cutoffHeight and nav.velocity[2] > -g.cutoffSpeed):
            self.cutoff = True
            self.lastThrottle = 0.0
            return Commands(throttle=0.0, gimbal=np.zeros(2))

        # 2. Coast until the braking altitude, holding upright with the RCS. Ignition is latched.
        if not self.ignited:
            if feetHeight > self.brakingAltitude(nav):
                self.lastThrottle = 0.0
                return self.tvc(nav, np.array([0.0, 0.0, 1.0]), t, engineOn=False, dynamicPressure=self.dynamicPressure(nav, t))
            self.ignited = True

        # 3. Guide the CG to cgHeight above the pad, so the feet land on it. Add the learned residual.
        aCmd = self.guidanceCommand(nav, t) + self.residual

        # 4. Lateral integral of position error, accumulated only low down, clamped against windup.
        dt = 0.0 if self.lastTime is None else max(t - self.lastTime, 0.0)
        self.lastTime = t
        if feetHeight <= g.integralHeight:
            self.lateralIntegral = np.clip(self.lateralIntegral - g.kIntegral * nav.position[0:2] * dt, -g.integralLimit, g.integralLimit)

        # 5. Terminal phase near the pad: rate-of-descent law vertically, PID position hold laterally.
        if feetHeight <= g.terminalHeight:
            # Alignment hold: do not descend the last metres while still sliding sideways or off the pad,
            # unless propellant is down to the floor, in which case landing a little off beats falling.
            misaligned = (g.alignSpeed > 0 and np.linalg.norm(nav.velocity[0:2]) > g.alignSpeed) or \
                         (g.alignOffset > 0 and np.linalg.norm(nav.position[0:2]) > g.alignOffset)
            fuelLeft = nav.propellantMass / self.config.spacecraft.propellant.initialMass > g.holdPropellantFloor
            if misaligned and fuelLeft:
                # Hold at alignHoldHeight rather than wherever the vehicle happens to be: a lean while hovering
                # at a few tens of centimetres puts a foot on the pad, which forces a cutoff mid-correction.
                descentRate = float(np.clip(g.kHold * (g.alignHoldHeight - feetHeight), -abs(g.descentRate), 0.5))
            else:
                descentRate = g.descentRate
            # The integral trims the believed-vs-real thrust gap: without it a weak engine cannot hover.
            self.verticalIntegral = float(np.clip(self.verticalIntegral + g.kDescentIntegral * (descentRate - nav.velocity[2]) * dt,
                                                  -g.descentIntegralLimit, g.descentIntegralLimit))
            if self.canHover(nav):
                vertical = (self.config.environment.gravity + g.kDescent * (descentRate - nav.velocity[2])
                            + self.verticalIntegral + self.residual[2])
            else:
                vf = abs(g.descentRate)
                vz = nav.velocity[2]
                stopping = max(vz * vz - vf * vf, 0.0) / (2.0 * max(feetHeight, 0.5)) if vz < 0.0 else 0.0
                vertical = self.config.environment.gravity + stopping + self.residual[2]
            lateral = -g.kPosition * nav.position[0:2] - g.kVelocity * nav.velocity[0:2] + self.lateralIntegral + self.residual[0:2]
        else:
            vertical = aCmd[2]
            lateral = aCmd[0:2]
        vertical = max(vertical, g.minVerticalAccel)            # the engine cannot push down
        lateral = lateral - g.windFeedforwardGain * self.windAccel   # lean against the wind the IMU says is pushing

        # 6. Limit the tilt of the thrust vector, more generously at altitude than near the ground.
        horizontal = np.linalg.norm(lateral)
        maxHorizontal = vertical * np.tan(self.tiltLimit(feetHeight))
        if horizontal > maxHorizontal:
            lateral = lateral * maxHorizontal / horizontal

        # Scale the whole command by the inverse thrust estimate: the TVC turns acceleration into throttle assuming
        # nominal thrust, so a 0.9 engine gets asked for 1/0.9 of the acceleration and delivers what was wanted.
        commands = self.tvc(nav, np.array([lateral[0], lateral[1], vertical]) / self.thrustScale, t, dynamicPressure=self.dynamicPressure(nav, t))
        self.lastThrottle = commands.throttle
        return commands
