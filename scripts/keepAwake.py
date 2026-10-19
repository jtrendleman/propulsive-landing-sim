""" keepAwake.py
Runs a command while telling Windows not to sleep, then exits with the command's exit code (launchDetached.py --awake).
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.0.0

Usage: python scripts/keepAwake.py <command ...>
"""

import ctypes
import subprocess
import sys

esContinuous = 0x80000000
esSystemRequired = 0x00000001


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/keepAwake.py <command ...>")
    ctypes.windll.kernel32.SetThreadExecutionState(esContinuous | esSystemRequired)
    try:
        code = subprocess.call(sys.argv[1:])
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(esContinuous)
    sys.exit(code)


if __name__ == "__main__":
    main()
