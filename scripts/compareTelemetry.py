""" compareTelemetry.py
Flies the nominal mission and overlays it on four real Falcon 9 RTLS landings from webcast telemetry.
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.0.0

Prints the landing-burn numbers the vehicle config is fitted to (docs/references.txt [D1], [D2]) and writes
telemetryComparison.png: speed against altitude over the last 6 km, and speed against time to standstill.
"""

import argparse
import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

import matplotlib.pyplot as plt
import numpy as np

from landingSim.analysis.monteCarlo import runEpisode
from landingSim.analysis.telemetry import landingBurn, loadTelemetry, missions, resample
from landingSim.config import loadConfig
from landingSim.gnc.controller import LandingController
from landingSim.outputs import outputDir


def main():
    parser = argparse.ArgumentParser(description="Simulated landing against real Falcon 9 RTLS telemetry.")
    parser.add_argument("--config", type=pathlib.Path, default=repoRoot / "config" / "earthBooster.toml", help="scenario file")
    parser.add_argument("--telemetry", type=pathlib.Path, default=repoRoot / "data" / "telemetry")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="default: outputs/<version folder>/telemetryComparison.png")
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    traj, result = runEpisode(cfg, LandingController(cfg), sensorSeed=0)
    simSpeed = np.linalg.norm(traj.states[:, 3:6], axis=1)
    simAltitude = traj.states[:, 2]
    stop = int(np.argmax(simSpeed < 1.0)) if np.any(simSpeed < 1.0) else len(traj.time) - 1
    ignition = int(np.argmax(traj.commands[:, 0] > 0.0))

    fig, (axAlt, axTime) = plt.subplots(1, 2, figsize=(13, 5.5))
    print(f"{'flight':12s} {'ignition alt':>13s} {'ignition speed':>15s} {'burn to stop':>13s} {'m/(Cd A)':>10s}")
    for name in missions:
        telemetry = loadTelemetry(args.telemetry, name)
        burn = landingBurn(telemetry)
        print(f"{name:12s} {burn.ignitionAltitude:10.0f} m {burn.ignitionSpeed:11.1f} m/s {burn.duration:10.1f} s {burn.ballisticCoefficient:7.0f} kg/m^2")
        t, v, h = resample(telemetry, burn.ignitionTime - 15.0, burn.stopTime + 1.0, step=0.5)
        axAlt.plot(v, h / 1000.0, linewidth=1.2, label=f"{name} (webcast)")
        axTime.plot(t - burn.stopTime, v, linewidth=1.2, label=f"{name} (webcast)")
    print(f"{'simulation':12s} {simAltitude[ignition]:10.0f} m {simSpeed[ignition]:11.1f} m/s {traj.time[stop] - traj.time[ignition]:10.1f} s"
          f"   ({result.outcome}, {result.finalOffset:.2f} m from the pad, {100 * result.propellantFraction:.0f} % propellant left)")

    window = simAltitude <= 6000.0
    axAlt.plot(simSpeed[window], simAltitude[window] / 1000.0, "k-", linewidth=2.2, label="simulation (nominal)")
    axAlt.plot(simSpeed[ignition], simAltitude[ignition] / 1000.0, "ko", markersize=6, label="simulated ignition")
    axTime.plot(traj.time[: stop + 1] - traj.time[stop], simSpeed[: stop + 1], "k-", linewidth=2.2, label="simulation (nominal)")
    axAlt.set(xlabel="speed (m/s)", ylabel="altitude (km)", title="Landing burn: speed against altitude", ylim=(0, 6.2))
    axTime.set(xlabel="time to standstill (s)", ylabel="speed (m/s)", title="Landing burn: speed against time", xlim=(-50, 2))
    for ax in (axAlt, axTime):
        ax.grid(True, color="0.8", linewidth=0.6)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Falcon 9-class booster against Falcon 9 RTLS webcast telemetry (altitude shown to 0.1 km)")
    fig.tight_layout()
    out = args.out if args.out is not None else outputDir(repoRoot) / "telemetryComparison.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
