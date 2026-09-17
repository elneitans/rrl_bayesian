"""Aggregate historical or corrected logs using the runner's results directory."""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def collect_training(results_dir, game, variant):
    base = Path(results_dir)
    base = base if base.is_absolute() else ROOT / base
    base = base / game if variant == "rrtl_legacy" else base / variant / game
    frames = []
    for model in ("logical", "comparative"):
        for path in sorted((base / f"{model}_version").rglob("*train.csv")):
            frame = pd.read_csv(path)
            frame = frame[["Run", "Iteration", "Episode", "Reward"]].copy()
            frame["Model version"] = model.capitalize()
            frame["Variant"] = variant
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No training CSV files found under {base}")
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Read results_dir, game and variant from runner JSON")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--game", choices=("Breakout", "Pong", "DemonAttack"))
    parser.add_argument("--variant", choices=("rrtl_legacy", "rrtl_corrected"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    game = args.game or config.get("game")
    games = [game] if game else ["Breakout", "Pong", "DemonAttack"]
    variant = args.variant or config.get("variant", "rrtl_legacy")
    results = args.results_dir or config.get("results_dir", "results")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for game in games:
        frame = collect_training(results, game, variant)
        output = args.output_dir / f"train_data_{variant}_{game}.csv"
        frame.to_csv(output, index=False)
        print(output)


if __name__ == "__main__":
    main()
