""" optimizeGuidance.py
Searches the guidance knobs for the least propellant that still meets the 95 % success rate, then confirms and writes the config.
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.1.0
"""

import argparse
import json
import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

from landingSim.analysis.monteCarlo import missionSuccessRate
from landingSim.analysis.optimize import evaluateCandidate, optimizeGuidance, selectConfirmed, withGuidance, writeConfig
from landingSim.config import loadConfig
from landingSim.outputs import outputDir


def main():
    parser = argparse.ArgumentParser(description="Minimise propellant subject to the mission success rate.")
    parser.add_argument("--episodes", type=int, default=40, help="draws per candidate during the search")
    parser.add_argument("--confirm-episodes", dest="confirmEpisodes", type=int, default=100)
    parser.add_argument("--max-evaluations", dest="maxEvaluations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=pathlib.Path, default=repoRoot / "config" / "earthBoosterOptimized.toml")
    parser.add_argument("--start", nargs="*", default=[], metavar="KNOB=VALUE",
                        help="starting point overrides, e.g. referenceDecel=3.5 (default: the config's values)")
    parser.add_argument("--top", type=int, default=5, help="how many in-sample candidates to confirm on fresh draws")
    parser.add_argument("--from-history", dest="fromHistory", type=pathlib.Path, default=None,
                        help="skip the search and select from a saved outputs/<version>/optimization.json")
    parser.add_argument("--config", type=pathlib.Path, default=repoRoot / "config" / "earthBooster.toml", help="scenario file")
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    starts = {k: float(v) for k, v in (item.split("=") for item in args.start)}
    cfg = withGuidance(cfg, **starts)
    seed = cfg.monteCarlo.seed if args.seed is None else args.seed
    # Search ranges bracket the values derived for the Falcon 9-class booster (see the config comments): braking from
    # a near-floor burn to a late hard one, the lateral lead and time-to-go floor around the loop-bandwidth values,
    # the terminal phase from where the lateral time-to-go floor bites to well above it, and the terminal gains
    # within a factor of about two of the bandwidth-derived ones.
    bounds = {"referenceDecel": (14.0, 24.0), "ignitionMargin": (0.0, 0.3), "lateralLead": (3.0, 15.0), "lateralTgoMin": (3.0, 10.0),
              "terminalHeight": (50.0, 250.0), "kPosition": (0.1, 0.6), "kVelocity": (0.4, 1.6)}
    bins = {k: (high - low) / 10 for k, (low, high) in bounds.items()}

    def report(k, changes, summary, value):
        knobs = "  ".join(f"{name}={v:6.3f}" for name, v in changes.items())
        print(f"  eval {k:3d}: {knobs}  used {100 * summary['meanPropellantUsed']:5.1f} %  "
              f"success {100 * summary['successRate']:5.1f} %  objective {value:.4f}", flush=True)

    if args.fromHistory is not None:
        saved = json.loads(args.fromHistory.read_text(encoding="utf-8"))
        history = [(h["changes"], {"successRate": h["successRate"], "meanPropellantUsed": h["meanPropellantUsed"]}, h["objective"]) for h in saved["history"]]
        print(f"selecting from {len(history)} saved candidates in {args.fromHistory}")
    else:
        print(f"searching {list(bounds)} from {starts or 'config values'} with {args.episodes} draws per candidate, seed {seed}")
        _, _, history = optimizeGuidance(cfg, bounds, args.episodes, seed, args.workers, args.maxEvaluations, progress=report)

    # Validated selection: the in-sample winner sits on the edge of the feasible region by construction,
    # so confirm the cheapest feasible candidates on fresh draws and keep the cheapest that still passes.
    print(f"\nconfirming the {args.top} cheapest in-sample feasible candidates on {args.confirmEpisodes} fresh draws (seed {seed + 1})")

    def reportConfirm(changes, summary):
        knobs = "  ".join(f"{name}={v:6.3f}" for name, v in changes.items())
        print(f"  {knobs}  used {100 * summary['meanPropellantUsed']:5.1f} %  success {100 * summary['successRate']:5.1f} %  "
              f"mission success: {'yes' if summary['missionSuccess'] else 'no'}", flush=True)

    best, confirmed, confirmations = selectConfirmed(cfg, history, args.confirmEpisodes, seed + 1, args.workers, top=args.top,
                                                     progress=reportConfirm, bins=bins)
    baseline = evaluateCandidate(loadConfig(args.config), args.confirmEpisodes, seed + 1, args.workers)

    def line(name, s):
        print(f"  {name:10s} used {100 * s['meanPropellantUsed']:5.1f} %  success {100 * s['successRate']:5.1f} %  "
              f"soft {100 * s['softFailRate']:4.1f} %  hard {100 * s['hardFailRate']:4.1f} %  mission success: {'yes' if s['missionSuccess'] else 'no'}")

    print()
    line("baseline", baseline)
    line("optimized", confirmed)
    if not confirmed["missionSuccess"]:
        print("  WARNING: no candidate confirmed the mission success rate; the one with the highest confirmed rate was written")
    chosen = {k: getattr(best.guidance, k) for k in bounds}
    print("  knobs     " + "  ".join(f"{k}={v:.3f}" for k, v in chosen.items()))

    header = (f"Optimized guidance: least propellant with >= {100 * missionSuccessRate:.0f} % Monte Carlo success.\n"
              f"Search: {len(history)} candidates x {args.episodes} draws (seed {seed}); validated on {args.confirmEpisodes} fresh draws (seed {seed + 1}): "
              f"used {100 * confirmed['meanPropellantUsed']:.1f} %, success {100 * confirmed['successRate']:.1f} %.\n"
              f"Generated by scripts/optimizeGuidance.py from {args.config.name}.")
    writeConfig(best, args.out, header=header)
    (outputDir(repoRoot) / "optimization.json").write_text(json.dumps({
        "bounds": bounds, "searchEpisodes": args.episodes, "seed": seed, "chosen": chosen,
        "confirmed": confirmed, "baseline": baseline,
        "confirmations": [{"changes": c, "summary": s} for c, s in confirmations],
        "history": [{"changes": c, "objective": v, "successRate": s["successRate"], "meanPropellantUsed": s["meanPropellantUsed"]} for c, s, v in history],
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out} and outputs/<version>/optimization.json")


if __name__ == "__main__":
    main()
