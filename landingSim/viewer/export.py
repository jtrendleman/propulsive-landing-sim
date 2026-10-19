""" export.py
Turns a Trajectory plus config into the JSON frames the HTML viewer animates, and writes the viewer file.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.1.0

The viewer is a consumer of the log only. Everything it shows is read or derived from the log here:
positions, quaternions, commands, propellant, tilt, load factor, and the zero-effort impact point
(where the vehicle would hit the ground if the engine cut now and it coasted under gravity).
"""

import json
import pathlib

import numpy as np

from landingSim.core import attitude
from landingSim.vehicle.mass import massProperties

templatePath = pathlib.Path(__file__).with_name("replayTemplate.html")
dataMarker = "/*__REPLAY_DATA__*/"


def zeroEffortImpact(position, velocity, feetHeight, gravity):
    """Ground point reached by coasting from here with the engine off; (x, y, timeToImpact)."""
    h = max(feetHeight, 0.0)
    vz = velocity[2]
    t = (vz + np.sqrt(vz**2 + 2 * gravity * h)) / gravity
    return float(position[0] + velocity[0] * t), float(position[1] + velocity[1] * t), float(t)


def replayData(trajectory, config, maxFrames=1500):
    """Frames downsampled to at most maxFrames, always keeping the last row."""
    n = len(trajectory.time)
    stride = max(1, int(np.ceil(n / maxFrames)))
    indices = list(range(0, n, stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    gravity = config.environment.gravity
    velocity = trajectory.states[:, 3:6]
    acceleration = np.gradient(velocity, trajectory.time, axis=0) if n > 1 else np.zeros_like(velocity)
    loadFactor = np.linalg.norm(acceleration, axis=1) / gravity

    frames = []
    for i in indices:
        s = trajectory.state(i)
        cgHeight = massProperties(s, config.spacecraft).cgHeight
        feetHeight = s.position[2] - cgHeight
        bodyZ = attitude.rotateBodyToWorld(s.quaternion, np.array([0.0, 0.0, 1.0]))
        zemX, zemY, zemT = zeroEffortImpact(s.position, s.velocity, feetHeight, gravity)
        frames.append({
            "t": float(trajectory.time[i]),
            "p": [float(v) for v in s.position],
            "v": [float(v) for v in s.velocity],
            "q": [float(v) for v in s.quaternion],                       # scalar-first [w, x, y, z]
            "throttle": float(trajectory.commands[i, 0]),
            "gimbal": [float(trajectory.commands[i, 1]), float(trajectory.commands[i, 2])],
            "propellant": float(s.propellantMass / config.spacecraft.propellant.initialMass),
            "cg": float(cgHeight),
            "feet": float(feetHeight),
            "speed": float(np.linalg.norm(s.velocity)),
            "vz": float(s.velocity[2]),
            "tilt": float(np.degrees(np.arccos(np.clip(bodyZ[2], -1.0, 1.0)))),
            "load": float(min(loadFactor[i], 20.0)),
            "zem": [zemX, zemY, zemT],
        })

    spacecraft = config.spacecraft
    return {
        "endReason": trajectory.endReason,
        "geometry": {
            "bodyLength": spacecraft.geometry.bodyLength,
            "bodyRadius": spacecraft.geometry.bodyRadius,
            "footRadius": spacecraft.landingGear.footRadius,
            "strutCount": spacecraft.landingGear.strutCount,
            "maxGimbalAngle": spacecraft.engine.maxGimbalAngle,
        },
        "environment": {
            "gravity": gravity,
            "windMeanSpeed": config.environment.wind.meanSpeed,
        },
        "engine": {
            "maxThrust": spacecraft.engine.maxThrust * spacecraft.engine.count,
            "minThrottle": spacecraft.engine.minThrottle,
        },
        "frames": frames,
    }


def writeReplay(trajectory, config, path, maxFrames=1500):
    """Writes a single HTML file with the replay data inlined; open it in any browser."""
    template = templatePath.read_text(encoding="utf-8")
    assert dataMarker in template, "replayTemplate.html is missing the data marker"
    payload = json.dumps(replayData(trajectory, config, maxFrames))
    path = pathlib.Path(path)
    path.write_text(template.replace(dataMarker, payload), encoding="utf-8")
    return path
