import pandas as pd
import pytest

from datasets.make_train_dataset_per_game import collect_training


@pytest.mark.parametrize("variant", ["rrtl_legacy", "rrtl_corrected"])
def test_aggregation_uses_configured_results_directory(tmp_path, variant):
    base = tmp_path if variant == "rrtl_legacy" else tmp_path / variant
    folder = base / "Breakout/comparative_version" / ("train" if variant == "rrtl_legacy" else "run_1")
    folder.mkdir(parents=True)
    name = "Breakout_run_1_train.csv" if variant == "rrtl_legacy" else "train.csv"
    pd.DataFrame({"Run": [1, 1], "Iteration": [1, 2], "Episode": [0, 0], "Reward": [0, 1]}).to_csv(folder / name)
    frame = collect_training(tmp_path, "Breakout", variant)
    assert len(frame) == 2
    assert frame["Variant"].unique().tolist() == [variant]
    assert frame["Model version"].unique().tolist() == ["Comparative"]
