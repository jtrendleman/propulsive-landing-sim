""" plots3d.py
MATLAB-style 3D figures: one landing trajectory with vehicle glyphs and a time-window montage, and a Monte Carlo dispersion surface.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0

Everything plotted is read or derived from trajectory logs; nothing here touches the physics.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from landingSim.core import attitude
from landingSim.vehicle.mass import massProperties


def styleAxes(ax, title, xLabel="X (m)", yLabel="Y (m)", zLabel="Z (m)"):
    """White panes, light grid, MATLAB-like labels."""
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel(xLabel, labelpad=6)
    ax.set_ylabel(yLabel, labelpad=6)
    ax.set_zlabel(zLabel, labelpad=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis._axinfo["grid"].update(color=(0.75, 0.75, 0.75, 1.0), linewidth=0.6, linestyle="-")
    ax.tick_params(labelsize=8)


def groundMesh(ax, halfWidth, spacing, z=0.0, color="#7b86c9", alpha=0.3):
    """Flat wireframe ground the way MATLAB's mesh() draws it."""
    xs = np.arange(-halfWidth, halfWidth + spacing, spacing)
    gx, gy = np.meshgrid(xs, xs)
    ax.plot_wireframe(gx, gy, np.full_like(gx, z), color=color, linewidth=0.4, alpha=alpha)


def padMarker(ax, radius=0.5, z=0.0):
    theta = np.linspace(0, 2 * np.pi, 100)
    for r, style in [(radius, "r-"), (4 * radius, "k--")]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), np.full_like(theta, z), style, linewidth=1.2)
    ax.plot([0], [0], [z], "r+", markersize=12, markeredgewidth=1.5)


def coloredPath(ax, points, values, cmap="jet", vmin=None, vmax=None, linewidth=2.0):
    """A 3D polyline with each segment colored by a scalar; returns the mappable for a colorbar."""
    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = colors.Normalize(np.min(values) if vmin is None else vmin, np.max(values) if vmax is None else vmax)
    collection = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth)
    collection.set_array(0.5 * (values[:-1] + values[1:]))
    ax.add_collection3d(collection)
    return collection


def vehicleGlyph(ax, state, commands, config, scale):
    """Body axis as a black bar, thrust vector as a red arrow, at the CG."""
    bodyZ = attitude.rotateBodyToWorld(state.quaternion, np.array([0.0, 0.0, 1.0]))
    base = state.position - bodyZ * scale * 0.5
    tip = state.position + bodyZ * scale * 0.5
    ax.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]], "k-", linewidth=2.5, solid_capstyle="round")
    ax.plot([tip[0]], [tip[1]], [tip[2]], "ko", markersize=3)
    throttle, gx, gy = commands[0], commands[1], commands[2]
    if throttle > 0.0:
        # Thrust force on the vehicle: along +z body tilted by the gimbal, applied at the base.
        thrustBody = attitude.normalize(np.array([np.tan(gx), np.tan(gy), 1.0]))
        thrustWorld = attitude.rotateBodyToWorld(state.quaternion, thrustBody)
        length = scale * (0.3 + 0.9 * throttle)
        ax.quiver(base[0], base[1], base[2], thrustWorld[0], thrustWorld[1], thrustWorld[2],
                  length=length, color="r", linewidth=1.3, arrow_length_ratio=0.12)


def trajectoryArrays(trajectory, config):
    n = len(trajectory.time)
    positions = trajectory.states[:, 0:3]
    speeds = np.linalg.norm(trajectory.states[:, 3:6], axis=1)
    cgHeights = np.array([massProperties(trajectory.state(i), config.spacecraft).cgHeight for i in range(n)])
    return positions, speeds, cgHeights


def plotLanding3d(trajectory, config, path, glyphEvery=4.0, title="Propulsive landing trajectory"):
    positions, speeds, _ = trajectoryArrays(trajectory, config)
    halfWidth = max(60.0, 1.3 * np.max(np.abs(positions[:, 0:2])))

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    styleAxes(ax, f"{title}  (ended: {trajectory.endReason} at t = {trajectory.time[-1]:.1f} s)")
    groundMesh(ax, halfWidth, halfWidth / 12)
    padMarker(ax)
    ax.plot(positions[:, 0], positions[:, 1], np.zeros(len(positions)), color="0.55", linewidth=1.0, label="ground track")
    mappable = coloredPath(ax, positions, speeds)
    for t in np.arange(0.0, trajectory.time[-1] + 1e-9, glyphEvery):
        i = int(np.searchsorted(trajectory.time, t))
        if i < len(trajectory.time):
            vehicleGlyph(ax, trajectory.state(i), trajectory.commands[i], config, scale=0.06 * positions[0, 2])
    ax.plot([], [], "k-", linewidth=2.5, label="body axis")
    ax.plot([], [], "r-", linewidth=1.5, label="thrust vector")
    ax.plot([], [], "r+", label="pad")
    ax.set_xlim(-halfWidth, halfWidth); ax.set_ylim(-halfWidth, halfWidth); ax.set_zlim(0, positions[0, 2] * 1.05)
    ax.set_box_aspect((1, 1, 1.3))
    ax.view_init(elev=22, azim=-55)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("speed (m/s)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def plotLandingMontage(trajectory, config, path, windows=None, title="Propulsive landing trajectory"):
    positions, speeds, _ = trajectoryArrays(trajectory, config)
    tEnd = trajectory.time[-1]
    if windows is None:
        edges = np.linspace(0.0, tEnd, 5)
        windows = [(edges[k], edges[k + 1]) for k in range(4)]
    norm = colors.Normalize(0.0, np.max(speeds))

    fig = plt.figure(figsize=(12, 9))
    for k, (t0, t1) in enumerate(windows):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        mask = (trajectory.time >= t0) & (trajectory.time <= t1)
        idx = np.where(mask)[0]
        if len(idx) < 2:
            idx = np.array([max(0, len(trajectory.time) - 2), len(trajectory.time) - 1])
        seg = positions[idx]
        styleAxes(ax, f"({k + 1}) t = {t0:.0f}-{t1:.0f} s")
        span = max(30.0, 1.4 * np.max(np.abs(seg[:, 0:2])))
        groundMesh(ax, span, span / 8)
        padMarker(ax)
        ax.plot(positions[:idx[0] + 1, 0], positions[:idx[0] + 1, 1], positions[:idx[0] + 1, 2], color="0.75", linewidth=0.8)
        coloredPath(ax, seg, speeds[idx], vmin=0.0, vmax=np.max(speeds), linewidth=2.5)
        ax.plot(seg[:, 0], seg[:, 1], np.zeros(len(seg)), color="0.55", linewidth=1.0)
        for j in np.linspace(idx[0], idx[-1], 3).astype(int):
            vehicleGlyph(ax, trajectory.state(j), trajectory.commands[j], config, scale=max(6.0, 0.12 * seg[0, 2]))
        zTop = max(seg[:, 2].max() * 1.15, 12.0)
        ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_zlim(0, zTop)
        ax.set_box_aspect((1, 1, 0.9))
        ax.view_init(elev=24, azim=-60)
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(left=0.02, right=0.86, top=0.92, bottom=0.04, wspace=0.05, hspace=0.12)
    cax = fig.add_axes([0.90, 0.25, 0.018, 0.5])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="jet"), cax=cax)
    cbar.set_label("speed (m/s)")
    fig.savefig(path, dpi=160)
    return fig


def landingDensitySurface(landingPoints, halfWidth, cells=81, sigma=None):
    """Gaussian-smoothed touchdown density on a grid; the surface the Monte Carlo bundle lands on."""
    xs = np.linspace(-halfWidth, halfWidth, cells)
    gx, gy = np.meshgrid(xs, xs)
    sigma = 0.04 * halfWidth if sigma is None else sigma
    z = np.zeros_like(gx)
    for px, py in landingPoints:
        z += np.exp(-((gx - px) ** 2 + (gy - py) ** 2) / (2 * sigma**2))
    return gx, gy, z / max(len(landingPoints), 1)


def plotDispersion3d(trajectories, results, config, path, title="Monte Carlo landing dispersion", footprintHalfWidth=10.0):
    """Left: the trajectory bundle over the pad. Right: touchdown density surface zoomed to the pad."""
    outcomeColors = {"succeeded": "#2ca02c", "softFailed": "#ff7f0e", "hardFailed": "#d62728"}
    finals = np.array([traj.states[-1, 0:2] for traj in trajectories])
    allXY = np.concatenate([traj.states[:, 0:2] for traj in trajectories])
    halfWidth = max(60.0, 1.1 * np.percentile(np.abs(allXY), 99))
    zTop = max(traj.states[0, 2] for traj in trajectories) * 1.05
    n = len(trajectories)
    counts = {k: sum(r.outcome == k for r in results) for k in outcomeColors}

    fig = plt.figure(figsize=(15, 7.5))
    fig.suptitle(f"{title}: {n} episodes, {100 * counts['succeeded'] / n:.0f} % succeeded", fontsize=13)

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    styleAxes(ax, "Trajectory bundle")
    groundMesh(ax, halfWidth, halfWidth / 12)
    padMarker(ax)
    for traj, result in zip(trajectories, results):
        p = traj.states[:, 0:3]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=outcomeColors[result.outcome], linewidth=0.9, alpha=0.85)
        ax.plot(p[:, 0], p[:, 1], np.zeros(len(p)), color="0.7", linewidth=0.5)
        ax.plot([p[-1, 0]], [p[-1, 1]], [0.0], "o", color=outcomeColors[result.outcome], markersize=3)
    for outcome, color in outcomeColors.items():
        ax.plot([], [], color=color, label=f"{outcome} ({counts[outcome]})")
    ax.set_xlim(-halfWidth, halfWidth); ax.set_ylim(-halfWidth, halfWidth); ax.set_zlim(0, zTop)
    ax.set_box_aspect((1, 1, 1.1))
    ax.view_init(elev=24, azim=-55)
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    styleAxes(ax2, f"Touchdown density within {footprintHalfWidth:.0f} m of the pad", zLabel="density")
    gx, gy, density = landingDensitySurface(finals, footprintHalfWidth, cells=121, sigma=0.5)
    density = density / (2 * np.pi * 0.5**2)                               # normalise each kernel to unit area
    ax2.plot_surface(gx, gy, density, cmap="jet", rstride=1, cstride=1, linewidth=0.15, edgecolor=(0, 0, 0, 0.12), antialiased=True)
    theta = np.linspace(0, 2 * np.pi, 100)
    zRing = np.full_like(theta, density.max() * 0.02)
    ax2.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), zRing, "w--", linewidth=1.5, label="0.5 m criterion")
    for outcome, color in outcomeColors.items():
        pts = finals[[r.outcome == outcome for r in results]]
        inside = pts[np.all(np.abs(pts) <= footprintHalfWidth, axis=1)]
        if len(inside):
            ax2.plot(inside[:, 0], inside[:, 1], np.full(len(inside), density.max() * 1.02), "o", color=color, markersize=3, label=outcome)
    ax2.set_xlim(-footprintHalfWidth, footprintHalfWidth); ax2.set_ylim(-footprintHalfWidth, footprintHalfWidth)
    ax2.set_box_aspect((1, 1, 0.6))
    ax2.view_init(elev=35, azim=-50)
    ax2.legend(loc="upper left", fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.04, wspace=0.08)
    fig.savefig(path, dpi=160)
    return fig


def animateLanding3d(trajectory, config, path, fps=20, speed=2.0, orbit=True, dpi=100, title="Propulsive landing trajectory"):
    """The 3D trajectory figure as a video: the path draws itself in, the glyph flies it, the camera orbits slowly.

    Writes MP4 through ffmpeg when it is installed, otherwise a GIF through Pillow. `speed` is
    playback speed relative to real time.
    """
    from matplotlib import animation

    positions, speeds, _ = trajectoryArrays(trajectory, config)
    halfWidth = max(60.0, 1.3 * np.max(np.abs(positions[:, 0:2])))
    tEnd = trajectory.time[-1]
    frameTimes = np.arange(0.0, tEnd + 1e-9, speed / fps)
    if frameTimes[-1] < tEnd:
        frameTimes = np.append(frameTimes, tEnd)
    holdFrames = int(1.5 * fps)
    frameTimes = np.append(frameTimes, np.full(holdFrames, tEnd))

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    styleAxes(ax, title)
    groundMesh(ax, halfWidth, halfWidth / 12)
    padMarker(ax)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="0.85", linewidth=0.8)      # full path, faint
    norm = colors.Normalize(0.0, np.max(speeds))
    segments = np.stack([positions[:-1], positions[1:]], axis=1)
    segmentSpeeds = 0.5 * (speeds[:-1] + speeds[1:])
    flown = Line3DCollection(segments[:1], cmap="jet", norm=norm, linewidth=2.2)
    flown.set_array(segmentSpeeds[:1])
    ax.add_collection3d(flown)
    groundTrack, = ax.plot([], [], [], color="0.55", linewidth=1.0)
    ax.set_xlim(-halfWidth, halfWidth); ax.set_ylim(-halfWidth, halfWidth); ax.set_zlim(0, positions[0, 2] * 1.05)
    ax.set_box_aspect((1, 1, 1.3))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="jet"), ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("speed (m/s)")
    ax.plot([], [], "k-", linewidth=2.5, label="body axis")
    ax.plot([], [], "r-", linewidth=1.5, label="thrust vector")
    ax.plot([], [], "r+", label="pad")
    ax.legend(loc="upper left", fontsize=8)
    readout = fig.text(0.02, 0.02, "", family="monospace", fontsize=9, va="bottom")
    glyphScale = 0.06 * positions[0, 2]
    glyphArtists = []

    def draw(k):
        t = frameTimes[k]
        i = min(int(np.searchsorted(trajectory.time, t)), len(trajectory.time) - 1)
        n = max(i, 1)
        flown.set_segments(segments[:n])
        flown.set_array(segmentSpeeds[:n])
        groundTrack.set_data(positions[:i + 1, 0], positions[:i + 1, 1])
        groundTrack.set_3d_properties(np.zeros(i + 1))
        for artist in glyphArtists:
            artist.remove()
        glyphArtists.clear()
        before = set(ax.lines) | set(ax.collections)
        vehicleGlyph(ax, trajectory.state(i), trajectory.commands[i], config, scale=glyphScale)
        glyphArtists.extend((set(ax.lines) | set(ax.collections)) - before)
        if orbit:
            ax.view_init(elev=22, azim=-55 + 40 * k / len(frameTimes))
        state = trajectory.state(i)
        feet = state.position[2] - massProperties(state, config.spacecraft).cgHeight
        readout.set_text(f"t = {t:6.2f} s   feet altitude = {feet:7.1f} m   speed = {speeds[i]:5.1f} m/s   "
                         f"throttle = {100 * trajectory.commands[i, 0]:5.1f} %   propellant = {100 * state.propellantMass / config.spacecraft.propellant.initialMass:5.1f} %"
                         + (f"   {trajectory.endReason}" if t >= tEnd else ""))
        return [flown, groundTrack, readout, *glyphArtists]

    anim = animation.FuncAnimation(fig, draw, frames=len(frameTimes), blit=False)
    path = pathlib.Path(path)
    if animation.writers.is_available("ffmpeg"):
        path = path.with_suffix(".mp4")
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=4000), dpi=dpi)
    else:
        path = path.with_suffix(".gif")
        anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
