""" runs.py
Folder convention for training runs: checkpoints/VX.Y.Z-MM-DD-YY/ holds model.zip, trainLog.txt, settings.json, and evaluation output.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.1.0
"""

import datetime
import json
import pathlib
import re

modelFile = "model.zip"
runPattern = re.compile(r"^V(\d+)\.(\d+)\.(\d+)-(\d{2})-(\d{2})-(\d{2})$")


def checkpointsRoot(repoRoot):
    return pathlib.Path(repoRoot) / "checkpoints"


def parseRunName(name):
    """'V1.2.3-10-17-26' -> ((1, 2, 3), date) or None if the folder does not follow the convention."""
    m = runPattern.match(name)
    if m is None:
        return None
    major, minor, patch, month, day, year = (int(g) for g in m.groups())
    return (major, minor, patch), datetime.date(2000 + year, month, day)


def allRunDirs(repoRoot):
    root = checkpointsRoot(repoRoot)
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir() and parseRunName(p.name) is not None]


def nextVersion(repoRoot):
    """Highest existing version with the patch number bumped; V1.0.0 when there are no runs yet."""
    versions = [parseRunName(p.name)[0] for p in allRunDirs(repoRoot)]
    if not versions:
        return (1, 0, 0)
    major, minor, patch = max(versions)
    return (major, minor, patch + 1)


def runName(version, date=None):
    date = datetime.date.today() if date is None else date
    return f"V{version[0]}.{version[1]}.{version[2]}-{date:%m-%d-%y}"


def newRunDir(repoRoot, version=None):
    """Create checkpoints/VX.Y.Z-MM-DD-YY. version is a 'X.Y.Z' string or None for the next patch number."""
    parsed = nextVersion(repoRoot) if version is None else tuple(int(v) for v in version.split("."))
    runDir = checkpointsRoot(repoRoot) / runName(parsed)
    runDir.mkdir(parents=True, exist_ok=False)
    return runDir


def listRuns(repoRoot):
    """Run folders that contain a saved model, lowest version first."""
    return sorted((p for p in allRunDirs(repoRoot) if (p / modelFile).exists()), key=lambda p: parseRunName(p.name))


def latestRun(repoRoot):
    runs = listRuns(repoRoot)
    if not runs:
        raise FileNotFoundError(f"no training run with a {modelFile} under {checkpointsRoot(repoRoot)}; run scripts/trainResidual.py first")
    return runs[-1]


def loadModel(modelPath, runDir=None):
    """Load a saved policy with the algorithm recorded in the run's settings.json (PPO if unknown)."""
    import json
    from stable_baselines3 import PPO, SAC, TD3
    settings = pathlib.Path(runDir or pathlib.Path(modelPath).parent) / "settings.json"
    algo = json.loads(settings.read_text(encoding="utf-8")).get("algo", "ppo") if settings.exists() else "ppo"
    return {"ppo": PPO, "sac": SAC, "td3": TD3}[algo].load(modelPath, device="cpu")


def resolveModel(repoRoot, target=None):
    """A run folder, a model zip, or None (latest run) -> (runDir, modelPath)."""
    if target is None:
        runDir = latestRun(repoRoot)
        return runDir, runDir / modelFile
    target = pathlib.Path(target)
    if target.is_dir():
        return target, target / modelFile
    return target.parent, target


def writeSettings(runDir, settings):
    (runDir / "settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
