"""Configurable corrected baseline. Historical CLI remains train_and_test.py."""

import argparse
from dataclasses import replace
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import pickle
import shutil
import subprocess

from bayesian_rrtl.config import BaselineConfig

ROOT = Path(__file__).resolve().parent


def runtime_manifest(config):
    versions = {dist.metadata["Name"]: dist.version for dist in metadata.distributions()}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True, check=False).stdout
    dot = shutil.which("dot")
    dot_version = None
    if dot:
        result = subprocess.run([dot, "-V"], capture_output=True, text=True, check=True)
        dot_version = (result.stderr or result.stdout).strip()
    sources = [*ROOT.glob("*.py"), *sorted((ROOT / "bayesian_rrtl").glob("*.py"))]
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sources}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "git_commit": revision, "git_dirty": bool(dirty),
            "source_hashes": source_hashes,
            "graphviz": dot_version, "config": config.to_dict(),
            "config_hash": config.config_hash, "resolved_env_id": config.resolved_env_id,
            "checkpoint_selection": "final", "interaction_unit": "agent env.step (startup excluded)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument("--checkpoint", type=Path,
                        help="Evaluate a trusted local checkpoint without training; use its saved config")
    parser.add_argument("--eval-split", choices=("validation", "test", "none"), default="validation")
    parser.add_argument("--run", type=int, help="Override run ID without overwriting existing results")
    parser.add_argument("--results-dir", help="Override the output root")
    args = parser.parse_args()
    if args.checkpoint and (args.run is not None or args.results_dir is not None or args.eval_split == "none"):
        parser.error("--checkpoint requires an evaluation split and cannot be combined with --run or --results-dir")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    from bayesian_rrtl.baseline import CorrectedRRLAgent
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
        with checkpoint.open("rb") as handle:
            payload = pickle.load(handle)
        config = BaselineConfig(**payload["config"])
        output = checkpoint.parent
        agent = CorrectedRRLAgent(config, output)
        agent.load_final_checkpoint(checkpoint)
        rows = agent.evaluate(getattr(config, f"{args.eval_split}_seeds"))
        destination = output / f"{args.eval_split}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(rows, indent=2))
        temporary.replace(destination)
        print(json.dumps({"checkpoint": str(checkpoint), "evaluation": str(destination),
                          "eval_split": args.eval_split, "episodes": len(rows)}, indent=2))
        return
    config = BaselineConfig.load(args.config or ROOT / "configs/breakout_smoke.json")
    overrides = {key: getattr(args, key) for key in ("run", "results_dir") if getattr(args, key) is not None}
    config = replace(config, **overrides)
    base = Path(config.results_dir)
    base = base if base.is_absolute() else ROOT / base
    output = base / config.variant / config.game / f"{config.model_version}_version" / f"run_{config.run}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = runtime_manifest(config)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    agent = CorrectedRRLAgent(config, output)
    summary = agent.train()
    if args.eval_split != "none":
        # Explicitly reload the final artifact: no training-return checkpoint selection.
        agent.load_final_checkpoint(output / "final_checkpoint.pkl")
        rows = agent.evaluate(getattr(config, f"{args.eval_split}_seeds"))
        (output / f"{args.eval_split}.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
