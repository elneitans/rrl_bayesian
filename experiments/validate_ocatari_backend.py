"""Strict real-Pong acceptance for plan steps 0/1; no training or test split.

Missing dependencies/ROMs fail this command. Output directories must be new.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from bayesian_rrtl.ocatari_backend import OCAtariBackend, COMPATIBILITY_ID, verified_versions


def source_manifest():
    versions = verified_versions()
    sources = {}
    for name in ("ocatari.core", "ocatari.ram.pong", "ocatari.ram.game_objects"):
        path = Path(importlib.import_module(name).__file__)
        sources[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    from ale_py.roms import get_rom_path
    rom = get_rom_path("pong")  # ALE checks the packaged ROM checksum itself.
    if rom is None:
        raise RuntimeError("Pong ROM unavailable; install the pinned official ale-py package")
    return {
        "executable": sys.executable, "python": platform.python_version(),
        "platform": platform.platform(), "isolated": sys.prefix != sys.base_prefix,
        "versions": versions, "compatibility_id": COMPATIBILITY_ID, "sources": sources,
        "rom": {"path": str(rom), "sha256": hashlib.sha256(rom.read_bytes()).hexdigest()},
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
    }


def check_truncation():
    backend = OCAtariBackend(max_episode_steps=1)
    try:
        initial = backend.reset(seed=41)
        step = backend.step(0)
        if (step.terminated, step.truncated) != (False, True):
            raise AssertionError("TimeLimit=1 must truncate, not terminate")
        return {"status": "passed", "initial_frame": asdict(initial), "step": asdict(step)}
    finally:
        backend.close()


def check_terminal(*, seed=41, max_steps=27000):
    """Bounded random rollout, independent of learning and any evaluation split."""
    backend = OCAtariBackend(max_episode_steps=max_steps)
    started = time.perf_counter()
    try:
        first = backend.reset(seed=seed)
        immutable_first = asdict(first)
        rng = np.random.default_rng(seed)
        action_counts = [0] * len(backend.actions)
        present = {obj.identity: 0 for obj in first.objects}
        positions = {obj.identity: set() for obj in first.objects}
        positive = negative = 0
        total = 0.0
        trajectory = hashlib.sha256()
        for count in range(1, max_steps + 1):
            action = int(rng.integers(len(backend.actions)))
            action_counts[action] += 1
            step = backend.step(action)
            total += step.raw_reward
            positive += int(step.raw_reward > 0)
            negative += int(step.raw_reward < 0)
            for obj in step.frame.objects:
                present[obj.identity] += int(obj.present)
                if obj.present:
                    positions[obj.identity].add(obj.bbox)
            trajectory.update(json.dumps(asdict(step), sort_keys=True).encode())
            if step.terminated or step.truncated:
                break
        if not step.terminated or step.truncated:
            raise AssertionError(f"No natural Pong terminal within {max_steps} random decisions")
        if not all(action_counts) or any(len(p) < 2 for p in positions.values()):
            raise AssertionError("Rollout did not exercise all actions and moving objects")
        if not positive + negative:
            raise AssertionError("No points observed")
        if asdict(first) != immutable_first:
            raise AssertionError("OCAtari mutated a previously returned ObjectFrame")
        return {
            "status": "passed", "seed": seed, "limit": max_steps, "steps": count,
            "actions": list(backend.actions), "action_counts": action_counts,
            "positive_points": positive, "negative_points": negative, "raw_return": total,
            "present_frames": present, "distinct_boxes": {k: len(v) for k, v in positions.items()},
            "terminated": step.terminated, "truncated": step.truncated,
            "initial_frame_unchanged": True, "final_frame": asdict(step.frame),
            "trajectory_sha256": trajectory.hexdigest(), "seconds": time.perf_counter() - started,
        }
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"scope": "ocatari_pong_plan_steps_0_1",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "command": [sys.executable, "-m", "experiments.validate_ocatari_backend",
                          "--output", str(args.output)],
              "not_run": ["relations_step_2", "training_cli_step_3", "exact_resume_step_4",
                          "training_smoke_step_5", "test_evaluation", "efficacy_campaign"]}
    try:
        result["manifest"] = source_manifest()
        check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
        result["pip_check"] = {"returncode": check.returncode, "stdout": check.stdout,
                               "stderr": check.stderr}
        check.check_returncode()
        result["truncation"] = check_truncation()
        result["terminal_rollout"] = check_terminal()
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (args.output / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
