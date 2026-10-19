""" optimizeAll.py
Runs the guidance optimizer, then the wind-schedule optimizer on its result: the full tuning pipeline in one command on any OS.
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.0.0

Usage: python scripts/optimizeAll.py   (or detached: python scripts/launchDetached.py --awake <log> -- python scripts/optimizeAll.py)
"""

import pathlib
import subprocess
import sys

scripts = pathlib.Path(__file__).resolve().parent


def main():
    for name in ("optimizeGuidance.py", "optimizeSchedule.py"):
        code = subprocess.call([sys.executable, str(scripts / name)])
        if code != 0:
            sys.exit(code)


if __name__ == "__main__":
    main()
