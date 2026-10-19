""" config.py
Loads one scenario TOML into frozen nested dataclasses, rejecting unknown or missing keys.
Author: Jack Rendleman  Date: 10-14-2026    Version: 1.0.0
"""
from dataclasses import dataclass
import dataclasses
import pathlib
import tomllib

class ConfigError(Exception):   pass
@dataclass(frozen=True)
class WindConfig:
    meanSpeed: float
    gustStd: float
    direction: float
    referenceHeight: float
    profileExponent: float
    turbulenceIntensity: float
    turbulenceLength: float
    verticalGustFraction: float
    directionShearPerKm: float
    boundaryLayerHeight: float

@dataclass(frozen=True)
class EnvironmentConfig:
    gravity: float
    seaLevelPressure: float
    seaLevelTemperature: float
    lapseRate: float
    gasConstant: float
    padSlope: float
    padSlopeDirection: float
    wind: WindConfig

@dataclass(frozen=True)
class MassConfig:
    dryMass: float
    cgHeightDry: float
    cgLateralOffset: tuple[float, float]
    inertiaDry: tuple[float, float, float]

@dataclass(frozen=True)
class TankConfig:
    density: float
    tankBottomHeight: float
    tankRadius: float
    domeRatio: float

@dataclass(frozen=True)
class PropellantConfig:
    initialMass: float
    mixtureRatio: float
    sloshEnabled: bool
    sloshDamping: float
    oxidizer: TankConfig
    fuel: TankConfig

@dataclass(frozen=True)
class EngineConfig:
    count: int
    maxThrust: float
    minThrottle: float
    isp: float
    maxGimbalAngle: float
    maxGimbalRate: float
    throttleTimeConstant: float
    ignitionDelay: float
    nozzleExitArea: float
    thrustMisalignment: tuple[float, float]

@dataclass(frozen=True)
class RcsConfig:
    thrusterCount: int
    thrustPerThruster: float
    momentArm: float

@dataclass(frozen=True)
class AeroConfig:
    dragCoefficient: float
    referenceArea: float
    gridFinArea: float
    cpHeight: float
    dragAlphaFactor: float
    cpHeightCrossflow: float
    finHeight: float
    finLiftSlope: float
    maxFinDeflection: float
    finMinDynamicPressure: float
    plumeDragFactor: float

@dataclass(frozen=True)
class LandingGearConfig:
    strutCount: int
    strutStroke: float
    strutCrossSection: float
    strutStiffness: float
    strutDamping: float
    footRadius: float
    footFriction: float
    frictionCoefficient: float
    tangentialStiffness: float
    strutHardening: float

@dataclass(frozen=True)
class GeometryConfig:
    bodyLength: float
    bodyRadius: float

@dataclass(frozen=True)
class StructureConfig:
    yieldStrength: float
    factorOfSafety: float
    loadFactorLimit: float

@dataclass(frozen=True)
class SpacecraftConfig:
    mass: MassConfig
    propellant: PropellantConfig
    engine: EngineConfig
    rcs: RcsConfig
    aero: AeroConfig
    landingGear: LandingGearConfig
    geometry: GeometryConfig
    structure: StructureConfig

@dataclass(frozen=True)
class MissionConfig:
    maxTime: float
    initialAltitude: float
    initialVerticalSpeed: float
    initialHorizontalOffset: float

@dataclass(frozen=True)
class MonteCarloConfig:
    episodes: int
    seed: int
    thrustStd: float
    dryMassStd: float
    cgOffsetStd: float
    initialOffsetStd: float
    randomWindDirection: bool
    cgLateralStd: float
    thrustMisalignmentStd: float
    padSlopeStd: float

@dataclass(frozen=True)
class RlConfig:
    decisionInterval: float
    maxResidualAccel: float

@dataclass(frozen=True)
class GuidanceConfig:
    coast: bool
    referenceDecel: float
    ignitionMargin: float
    tgoMin: float
    tgoMax: float
    lateralLead: float
    lateralTgoMin: float
    terminalHeight: float
    descentRate: float
    alignSpeed: float
    alignOffset: float
    holdPropellantFloor: float
    alignHoldHeight: float
    kHold: float
    kDescent: float
    kDescentIntegral: float
    descentIntegralLimit: float
    kPosition: float
    kVelocity: float
    kIntegral: float
    integralHeight: float
    integralLimit: float
    minVerticalAccel: float
    maxTilt: float
    highTilt: float
    tiltAltitude: float
    cutoffHeight: float
    cutoffSpeed: float
    kp: float
    kd: float
    windFeedforwardGain: float

@dataclass(frozen=True)
class WindScheduleConfig:
    enabled: bool
    averageWindSpeed: float
    strongWindSpeed: float
    minAirspeed: float
    padRadius: float
    referenceDecel: float
    ignitionMargin: float
    lateralLead: float
    maxTilt: float
    highTilt: float
    windFeedforwardGain: float
    kPosition: float
    kVelocity: float

@dataclass(frozen=True)
class SensorsConfig:
    enabled: bool
    gpsRate: float
    gpsLatency: float
    gpsPositionNoise: float
    gpsVelocityNoise: float
    altimeterRange: float
    altimeterNoise: float
    altimeterNoiseFraction: float
    accelNoise: float
    accelBias: float
    gyroNoise: float
    gyroBias: float
    alignmentError: float
    chamberPressureNoise: float
    chamberPressureBias: float
    propellantGaugeNoise: float
    propellantGaugeBias: float
    gimbalSensorNoise: float
    forecastSpeedError: float
    forecastDirectionError: float
    thrustEstimateTimeConstant: float
    engineEstimateTimeConstant: float
    windObserverTimeConstant: float

@dataclass(frozen=True)
class Config:
    environment: EnvironmentConfig
    spacecraft: SpacecraftConfig
    mission: MissionConfig
    monteCarlo: MonteCarloConfig
    rl: RlConfig
    guidance: GuidanceConfig
    windSchedule: WindScheduleConfig
    sensors: SensorsConfig


def _build(cls, table, pathForErrors):
    expected = {f.name for f in dataclasses.fields(cls)}
    given = set(table.keys())

    if given - expected:
        raise ConfigError(f"Unexpected keys in {pathForErrors}: {given - expected}")
    if expected - given:
        raise ConfigError(f"Missing keys in {pathForErrors}: {expected - given}")

    kwargs = {}
    for field in dataclasses.fields(cls):
        value = table[field.name]
        if dataclasses.is_dataclass(field.type):
            value = _build(field.type, value, f"{pathForErrors}.{field.name}")
        elif isinstance(value, list):
            value = tuple(value)
        kwargs[field.name] = value
    return cls(**kwargs)

def loadConfig(path) -> Config:
    path = pathlib.Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return _build(Config, raw, "config")

