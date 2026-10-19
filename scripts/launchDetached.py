""" launchDetached.py
Starts a command in its own session with output to a log file, so closing the terminal or restarting the editor does not stop it.
Author: Jack Rendleman  Date: 10-19-2026    Version: 2.0.0

Usage: python scripts/launchDetached.py [--awake] <logFile> -- <command ...>
--awake keeps the computer from sleeping while the command runs (caffeinate on macOS, the Windows power API on
Windows). Prints the process id; stop the run later with `kill <pid>` (macOS/Linux) or `taskkill /PID <pid> /T /F` (Windows).
"""

import os
import pathlib
import subprocess
import sys


def main():
    args = sys.argv[1:]
    awake = args[:1] == ["--awake"]
    if awake:
        args = args[1:]
    if "--" not in args or args.index("--") != 1:
        sys.exit("usage: python scripts/launchDetached.py [--awake] <logFile> -- <command ...>")
    logFile, command = args[0], args[2:]
    if awake:
        if os.name == "nt":
            command = [sys.executable, str(pathlib.Path(__file__).with_name("keepAwake.py"))] + command
        elif sys.platform == "darwin":
            command = ["caffeinate", "-is"] + command
    with open(logFile, "a", encoding="utf-8") as log:
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
        else:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(process.pid)


if __name__ == "__main__":
    main()
