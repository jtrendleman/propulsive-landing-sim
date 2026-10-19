""" trainResidual.py
Trains the residual policy (PPO, SAC, or TD3) on randomized landings; each run gets its own folder under checkpoints/.
Author: Jack Rendleman  Date: 10-18-2026    Version: 1.3.0

Long runs: --hours stops on wall clock, --checkpoint-every saves intermediate models under checkpoints/<run>/steps/,
and --eval-every flies a fixed held-out Monte Carlo set with the deterministic policy, keeping the best one in
checkpoints/<run>/best/model.zip. The final report is written for the best model when one exists.
"""

import argparse
import dataclasses
import json
import pathlib
import sys
import time

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

from stable_baselines3 import PPO, SAC, TD3
import shutil

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from landingSim.config import loadConfig
from landingSim.rl import landingEnv
from landingSim.rl.landingEnv import LandingEnv
from landingSim.analysis.monteCarlo import runMonteCarlo, summarize
from landingSim.rl.report import ResidualControllerFactory, plotTrainingCurve, writeRunReport
from landingSim.rl.runs import modelFile, newRunDir, writeSettings


class RunLog:
    """Prints a line and appends it to the run's trainLog.txt."""

    def __init__(self, path):
        self.path = path

    def __call__(self, line):
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class WallClockStop(BaseCallback):
    """Ends learning after a number of hours."""

    def __init__(self, hours):
        super().__init__()
        self.deadline = None if hours is None else time.time() + hours * 3600.0

    def _on_step(self):
        return self.deadline is None or time.time() < self.deadline


class HeldOutEvaluator(BaseCallback):
    """Every evalEvery steps: fly `episodes` fixed Monte Carlo draws with the deterministic policy, keep the best model."""

    def __init__(self, cfg, runDir, log, evalEvery, episodes, seed, workers):
        super().__init__()
        self.cfg, self.runDir, self.log = cfg, runDir, log
        self.evalEvery, self.episodes, self.seed, self.workers = evalEvery, episodes, seed, workers
        self.nextEval = evalEvery
        self.best = None
        self.history = []

    def _on_training_start(self):
        self.nextEval = self.num_timesteps + self.evalEvery          # a resumed run continues the schedule from where it is

    def _on_step(self):
        if self.evalEvery <= 0 or self.num_timesteps < self.nextEval:
            return True
        self.nextEval += self.evalEvery
        candidate = self.runDir / "steps" / "candidate"
        candidate.parent.mkdir(exist_ok=True)
        self.model.save(candidate)
        summary = summarize(runMonteCarlo(self.cfg, ResidualControllerFactory(str(candidate) + ".zip", self.runDir),
                                          episodes=self.episodes, seed=self.seed, workers=self.workers))
        score = (summary["successRate"], -summary["meanPropellantUsed"])
        self.history.append({"timesteps": self.num_timesteps, "successRate": summary["successRate"], "meanPropellantUsed": summary["meanPropellantUsed"]})
        improved = self.best is None or score > self.best
        if improved:
            self.best = score
            (self.runDir / "best").mkdir(exist_ok=True)
            shutil.copy(str(candidate) + ".zip", self.runDir / "best" / modelFile)
        self.log(f"  eval @ {self.num_timesteps:9d} steps: {100 * summary['successRate']:5.1f} % succeeded, "
                 f"{100 * summary['meanPropellantUsed']:5.1f} % propellant used on {self.episodes} held-out draws{'  <- best' if improved else ''}")
        (self.runDir / "evalHistory.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        return True


class OutcomeLogger(BaseCallback):
    """Logs a rolling success rate from the episode results the env puts in info, and keeps every episode for the training curve."""

    def __init__(self, log, window=50):
        super().__init__()
        self.log = log
        self.window = window
        self.outcomes = []
        self.episodes = []                      # (timestep, outcome, episode reward)
        self.startTime = time.time()

    def _on_step(self):
        for info in self.locals["infos"]:
            if "result" in info:
                self.outcomes.append(info["result"].outcome)
                self.episodes.append((self.num_timesteps, info["result"].outcome, float(info.get("episode", {}).get("r", float("nan")))))
                if len(self.outcomes) % self.window == 0:
                    recent = self.outcomes[-self.window:]
                    self.log(f"  {self.num_timesteps:8d} steps  {len(self.outcomes):5d} episodes  "
                             f"last {self.window}: {100 * recent.count('succeeded') / self.window:5.1f} % succeeded  "
                             f"{100 * recent.count('hardFailed') / self.window:5.1f} % hard failed  "
                             f"({time.time() - self.startTime:6.0f} s)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Train the PPO residual on randomized landings.")
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version", type=str, default=None, help="run version X.Y.Z; default bumps the patch of the highest existing run")
    parser.add_argument("--name", type=str, default="", help="free-text label stored in settings.json")
    parser.add_argument("--algo", choices=["ppo", "sac", "td3"], default="ppo",
                        help="PPO is on-policy and simple; SAC and TD3 are off-policy, usually more sample-efficient on continuous control")
    parser.add_argument("--learning-rate", dest="learningRate", type=float, default=1e-4)
    parser.add_argument("--log-std-init", dest="logStdInit", type=float, default=-1.0)
    parser.add_argument("--evaluate-episodes", dest="evaluateEpisodes", type=int, default=50, help="Monte Carlo draws for the end-of-run report (0 skips the report)")
    parser.add_argument("--workers", type=int, default=8, help="worker processes for the end-of-run Monte Carlo")
    parser.add_argument("--config", type=pathlib.Path, default=repoRoot / "config" / "earthBooster.toml", help="scenario file")
    parser.add_argument("--hours", type=float, default=None, help="stop after this many hours of wall clock (timesteps becomes an upper bound)")
    parser.add_argument("--checkpoint-every", dest="checkpointEvery", type=int, default=500_000, help="save an intermediate model this often (0 disables)")
    parser.add_argument("--eval-every", dest="evalEvery", type=int, default=500_000, help="held-out deterministic evaluation this often, keeps best/model.zip (0 disables)")
    parser.add_argument("--eval-episodes", dest="evalEpisodes", type=int, default=20)
    parser.add_argument("--eval-seed", dest="evalSeed", type=int, default=1)
    parser.add_argument("--resume", type=pathlib.Path, default=None,
                        help="continue training from a saved model zip (e.g. an earlier run's steps/ or best/ checkpoint); SAC/TD3 start with an empty replay buffer")
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    runDir = newRunDir(repoRoot, args.version)
    log = RunLog(runDir / "trainLog.txt")

    if args.algo == "ppo":
        algoKwargs = dict(n_steps=2048, batch_size=256, learning_rate=args.learningRate, gamma=0.995, gae_lambda=0.95,
                          ent_coef=0.0, clip_range=0.1, policy_kwargs=dict(log_std_init=args.logStdInit, net_arch=[64, 64]))
    elif args.algo == "sac":
        algoKwargs = dict(learning_rate=args.learningRate, gamma=0.995, batch_size=256, buffer_size=300_000,
                          learning_starts=5_000, train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[64, 64]))
    else:
        algoKwargs = dict(learning_rate=args.learningRate, gamma=0.995, batch_size=256, buffer_size=300_000,
                          learning_starts=5_000, train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[64, 64]))
    writeSettings(runDir, {
        "run": runDir.name, "name": args.name, "algo": args.algo, "config": str(args.config), "hours": args.hours,
        "resumedFrom": None if args.resume is None else str(args.resume),
        "checkpointEvery": args.checkpointEvery, "evalEvery": args.evalEvery, "evalEpisodes": args.evalEpisodes, "evalSeed": args.evalSeed, "timesteps": args.timesteps, "envs": args.envs, "seed": args.seed, "algoKwargs": algoKwargs,
        "rl": dataclasses.asdict(cfg.rl), "monteCarlo": dataclasses.asdict(cfg.monteCarlo),
        "wind": dataclasses.asdict(cfg.environment.wind),
        "reward": {k: getattr(landingEnv, k) for k in ["rewardScale", "crashReward", "landedReward", "offsetPenaltyPerMetre",
                                                       "criterionPenalty", "offsetPotentialPerMetre", "effortPenaltyPerSecond"]},
    })

    vecEnv = make_vec_env(lambda: LandingEnv(cfg), n_envs=args.envs, seed=args.seed,
                          vec_env_cls=SubprocVecEnv if args.envs > 1 else DummyVecEnv)
    # PPO: a small initial log-std keeps early episodes flying like the baseline, so it learns corrections
    # from there instead of first having to undo its own exploration noise. SAC/TD3 explore through their
    # own noise models and replay past episodes, so they need far fewer fresh landings per update.
    algorithm = {"ppo": PPO, "sac": SAC, "td3": TD3}[args.algo]
    if args.resume is not None:
        model = algorithm.load(args.resume, env=vecEnv, device="cpu")
        log(f"resumed from {args.resume} at {model.num_timesteps} steps")
    else:
        model = algorithm("MlpPolicy", vecEnv, seed=args.seed, device="cpu", verbose=0, **algoKwargs)
    log(f"run folder {runDir}")
    log(f"{args.algo.upper()}: {args.timesteps} timesteps on {args.envs} envs, residual authority {cfg.rl.maxResidualAccel} m/s^2")
    logger = OutcomeLogger(log)
    evaluator = HeldOutEvaluator(cfg, runDir, log, args.evalEvery, args.evalEpisodes, args.evalSeed, args.workers)
    callbacks = [logger, evaluator, WallClockStop(args.hours)]
    if args.checkpointEvery > 0:
        callbacks.append(CheckpointCallback(save_freq=max(args.checkpointEvery // max(args.envs, 1), 1), save_path=str(runDir / "steps"), name_prefix="model"))
    if args.hours is not None:
        log(f"wall-clock limit {args.hours} h; timesteps {args.timesteps} is an upper bound")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=args.resume is None)
    model.save(runDir / modelFile)
    log(f"saved final model {runDir / modelFile} after {model.num_timesteps} steps")
    vecEnv.close()
    plotTrainingCurve(logger.episodes, runDir / "trainingCurve.png")
    if (runDir / "best" / modelFile).exists():
        log(f"best held-out model kept at {runDir / 'best' / modelFile}; the report below is for it")
        shutil.copy(runDir / "best" / modelFile, runDir / modelFile)
    if args.evaluateEpisodes > 0:
        log("writing the run report")
        writeRunReport(runDir, cfg, evaluateEpisodes=args.evaluateEpisodes, workers=args.workers, log=log)
    log(f"run folder: {runDir}")


if __name__ == "__main__":
    main()
