""" evaluateResidual.py
Regenerates a training run's report: Monte Carlo baseline vs residual, dispersion, nominal comparison, summary.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.2.0
"""

import argparse
import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

from landingSim.config import loadConfig
from landingSim.rl.report import writeRunReport
from landingSim.rl.runs import resolveModel


def main():
    parser = argparse.ArgumentParser(description="Write or refresh the report in a training run's folder.")
    parser.add_argument("run", nargs="?", default=None, help="run folder or model zip; default: highest version under checkpoints/")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dispersion-episodes", dest="dispersionEpisodes", type=int, default=24)
    parser.add_argument("--config", type=pathlib.Path, default=repoRoot / "config" / "earthBooster.toml", help="scenario file")
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    runDir, _ = resolveModel(repoRoot, args.run)
    print(f"report for {runDir}")
    writeRunReport(runDir, cfg, evaluateEpisodes=args.episodes, seed=args.seed, workers=args.workers, dispersionEpisodes=args.dispersionEpisodes)
    print(f"wrote summary.txt, evaluation.json, monteCarloBaseline.png, monteCarloResidual.png, dispersion3d.png, nominalComparison.png, landingResidual3d.png to {runDir}")


if __name__ == "__main__":
    main()
