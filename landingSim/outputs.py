""" outputs.py
Folder convention for analysis outputs: outputs/VX.Y.Z-MM-DD-YY/, one folder per simulator version and day, like the training runs.
Author: Jack Rendleman  Date: 10-18-2026    Version: 1.0.0
"""

import datetime
import pathlib

from landingSim import version as simulatorVersion


def outputsRoot(repoRoot):
    return pathlib.Path(repoRoot) / "outputs"


def outputDir(repoRoot, version=None, date=None):
    """Today's folder for this simulator version, created if needed."""
    date = datetime.date.today() if date is None else date
    folder = outputsRoot(repoRoot) / f"V{version or simulatorVersion}-{date:%m-%d-%y}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def findLatest(repoRoot, filename):
    """The newest outputs/ folder that contains filename (by modification time), or today's folder if none does."""
    root = outputsRoot(repoRoot)
    candidates = sorted(root.glob(f"V*/{filename}"), key=lambda p: p.stat().st_mtime) if root.exists() else []
    return candidates[-1] if candidates else outputDir(repoRoot) / filename
