""" optimizeSchedule.py
Tunes the strong-wind end of the wind schedule: least propellant with a 95 % success rate in KSC strong wind, then checks average and extreme wind.
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.1.0

The average-wind end is the fuel-optimized [guidance] of earthBoosterOptimized.toml. The search flies KSC strong
draws with the schedule on, confirms the cheapest candidates on fresh draws, then checks that the blend did not
cost the average case its 95 % (it should not: at average wind the blend factor is near 0) and reports extreme wind.
"""

import argparse
import dataclasses
import json
import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

from landingSim.analysis.monteCarlo import missionSuccessRate
from landingSim.analysis.optimize import evaluateCandidate, optimizeGuidance, selectConfirmed, withSchedule, writeConfig
from landingSim.config import loadConfig
from landingSim.outputs import outputDir

bounds = {"referenceDecel": (14.0, 24.0), "ignitionMargin": (0.0, 0.3), "lateralLead": (3.0, 15.0),
          "maxTilt": (0.1, 0.4), "highTilt": (0.2, 0.7), "windFeedforwardGain": (0.5, 2.0),
          "kPosition": (0.1, 0.6), "kVelocity": (0.4, 1.6)}


def inScenario(scheduledConfig, scenarioPath):
    """The scheduled flight software flown in another scenario's wind (the vehicle and mission are shared)."""
    scenario = loadConfig(scenarioPath)
    return dataclasses.replace(scenario, guidance=scheduledConfig.guidance, windSchedule=scheduledConfig.windSchedule)


def line(name, s):
    print(f"  {name:28s} used {100 * s['meanPropellantUsed']:5.1f} %  success {100 * s['successRate']:5.1f} %  "
          f"soft {100 * s['softFailRate']:4.1f} %  hard {100 * s['hardFailRate']:4.1f} %  mission success: {'yes' if s['missionSuccess'] else 'no'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Tune the strong-wind end of the wind schedule.")
    parser.add_argument("--episodes", type=int, default=40, help="draws per candidate during the search")
    parser.add_argument("--confirm-episodes", dest="confirmEpisodes", type=int, default=100)
    parser.add_argument("--max-evaluations", dest="maxEvaluations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top", type=int, default=5, help="how many in-sample candidates to confirm on fresh draws")
    parser.add_argument("--from-history", dest="fromHistory", type=pathlib.Path, default=None,
                        help="skip the search and select from a saved outputs/<version>/scheduleOptimization.json")
    parser.add_argument("--base", type=pathlib.Path, default=repoRoot / "config" / "earthBoosterOptimized.toml", help="average-wind flight software")
    parser.add_argument("--out", type=pathlib.Path, default=repoRoot / "config" / "earthBoosterScheduled.toml")
    args = parser.parse_args()

    base = loadConfig(args.base)
    strongPath = repoRoot / "config" / "earthBoosterKscStrong.toml"
    cfg = inScenario(withSchedule(base), strongPath)
    seed = cfg.monteCarlo.seed if args.seed is None else args.seed
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
        print(f"searching the strong-wind knobs {list(bounds)} in KSC strong wind, {args.episodes} draws per candidate, seed {seed}")
        _, _, history = optimizeGuidance(cfg, bounds, args.episodes, seed, args.workers, args.maxEvaluations, progress=report, table="windSchedule")

    print(f"\nconfirming the {args.top} cheapest in-sample candidates on {args.confirmEpisodes} fresh strong-wind draws (seed {seed + 1})")

    def reportConfirm(changes, summary):
        knobs = "  ".join(f"{name}={v:6.3f}" for name, v in changes.items())
        print(f"  {knobs}  used {100 * summary['meanPropellantUsed']:5.1f} %  success {100 * summary['successRate']:5.1f} %", flush=True)

    best, confirmed, confirmations = selectConfirmed(cfg, history, args.confirmEpisodes, seed + 1, args.workers, top=args.top,
                                                     progress=reportConfirm, table="windSchedule", bins=bins)

    print("\nheld-out checks (100 draws each):")
    results = {}
    unscheduled = inScenario(base, strongPath)
    for name, config, s in [("KSC strong, no schedule", unscheduled, seed + 1), ("KSC strong, scheduled", best, seed + 1),
                            ("KSC strong, scheduled", best, seed + 2),
                            ("KSC average, no schedule", base, seed + 1), ("KSC average, scheduled", inScenario(best, args.base), seed + 1),
                            ("KSC average, no schedule", base, seed + 2), ("KSC average, scheduled", inScenario(best, args.base), seed + 2),
                            ("KSC extreme, no schedule", inScenario(base, repoRoot / "config" / "earthBoosterKscExtreme.toml"), seed + 1),
                            ("KSC extreme, scheduled", inScenario(best, repoRoot / "config" / "earthBoosterKscExtreme.toml"), seed + 1)]:
        summary = evaluateCandidate(config, args.confirmEpisodes, s, args.workers)
        results[f"{name}, seed {s}"] = summary
        line(f"{name}, seed {s}", summary)

    chosen = {k: getattr(best.windSchedule, k) for k in bounds}
    print("  strong-wind knobs  " + "  ".join(f"{k}={v:.3f}" for k, v in chosen.items()))
    scheduled = dataclasses.replace(base, windSchedule=best.windSchedule)
    header = (f"Wind-scheduled guidance: the fuel-optimized average-wind knobs of {args.base.name} blended toward strong-wind knobs\n"
              f"tuned for least propellant with >= {100 * missionSuccessRate:.0f} % success in KSC strong wind.\n"
              f"Search: {len(history)} candidates x {args.episodes} strong-wind draws (seed {seed}); "
              f"confirmed on {args.confirmEpisodes} fresh draws (seed {seed + 1}): success {100 * confirmed['successRate']:.1f} %.\n"
              f"Generated by scripts/optimizeSchedule.py.")
    writeConfig(scheduled, args.out, header=header)
    (outputDir(repoRoot) / "scheduleOptimization.json").write_text(json.dumps({
        "bounds": bounds, "searchEpisodes": args.episodes, "seed": seed, "chosen": chosen, "heldOut": results,
        "confirmations": [{"changes": c, "summary": s} for c, s in confirmations],
        "history": [{"changes": c, "objective": v, "successRate": s["successRate"], "meanPropellantUsed": s["meanPropellantUsed"]} for c, s, v in history],
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out} and outputs/<version>/scheduleOptimization.json")


if __name__ == "__main__":
    main()
