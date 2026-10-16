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

@dataclass(frozen=True)
class EnvironmentConfig:
    gravity: float
    seaLevelDensity: float
    scaleHeight: float
    wind: WindConfig

@dataclass(frozen=True)
class MassConfig:
    dryMass: float
    cgHeightDry: float
    inertiaDry: tuple[float, float, float]

@dataclass(frozen=True)
class PropellantConfig:
    initialMass: float
    density: float
    tankBottomHeight: float
    tankRadius: float

@dataclass(frozen=True)
class EngineConfig:
    count: int
    maxThrust: float
    minThrottle: float
    isp: float
    maxGimbalAngle: float
    maxGimbalRate: float

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

@dataclass(frozen=True)
class LandingGearConfig:
    strutCount: int
    strutStroke: float
    strutCrossSection: float
    strutStiffness: float
    strutDamping: float

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
    structure: StructureConfig

@dataclass(frozen=True)
class MissionConfig:
    maxTime: float
    initialAltitude: float
    initialVerticalSpeed: float
    initialHorizontalOffset: float

@dataclass(frozen=True)
class Config:
    environment: EnvironmentConfig
    spacecraft: SpacecraftConfig
    mission: MissionConfig


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

