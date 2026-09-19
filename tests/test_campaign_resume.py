"""Interrupted test campaigns preserve completed runs and publish atomically."""
import json
from types import SimpleNamespace

import pytest

from experiments import campaign_h7 as campaign


def rows():
    return [{"seed": 101, "return": 0.0, "steps": 2,
             "terminated": False, "truncated": True}]


@pytest.fixture
def pending(tmp_path, monkeypatch):
    protocol = {"games": ["Breakout"], "variants": ["batch_tree"],
                "training_seeds": [1, 2, 3], "test_seeds": [101]}
    paths = [tmp_path / "Breakout" / "batch_tree" / str(seed) / "test.json"
             for seed in protocol["training_seeds"]]
    for path in paths:
        path.parent.mkdir(parents=True)
    monkeypatch.setattr(campaign, "settings", lambda *args: (
        None, SimpleNamespace(test_seeds=[101])))
    calls = []

    def load(path, **kwargs):
        calls.append(int(path.parent.name))
        return SimpleNamespace(evaluate=lambda *args: rows()), None

    monkeypatch.setattr(campaign, "load_checkpoint", load)
    return protocol, tmp_path, paths, calls, load


def test_resume_after_one_of_three_results(pending, monkeypatch):
    protocol, output, paths, calls, load = pending

    def interrupted(path, **kwargs):
        if path.parent.name == "2":
            raise RuntimeError("simulated interruption")
        return load(path, **kwargs)

    monkeypatch.setattr(campaign, "load_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        campaign.evaluate_test(protocol, output)
    assert [p.exists() for p in paths] == [True, False, False]
    saved = paths[0].read_bytes(), paths[0].stat().st_mtime_ns
    monkeypatch.setattr(campaign, "load_checkpoint", load)
    campaign.evaluate_test(protocol, output)
    assert calls == [1, 2, 3]
    assert (paths[0].read_bytes(), paths[0].stat().st_mtime_ns) == saved
    assert all(json.loads(p.read_text()) == rows() for p in paths)
    campaign.evaluate_test(protocol, output)
    assert calls == [1, 2, 3]


@pytest.mark.parametrize("contents", [
    "{", "[]", json.dumps(rows() * 2),
    json.dumps([{**rows()[0], "seed": 102}]),
    json.dumps([{**rows()[0], "return": float("nan")}]),
    json.dumps([{**rows()[0], "steps": 0}]),
    json.dumps([{**rows()[0], "truncated": False}]),
    json.dumps([{**rows()[0], "terminated": 1}]),
])
def test_invalid_existing_result_is_preserved_before_evaluation(pending, contents):
    protocol, output, paths, calls, _ = pending
    paths[-1].write_text(contents)
    with pytest.raises(ValueError, match=str(paths[-1])):
        campaign.evaluate_test(protocol, output)
    assert calls == []
    assert paths[-1].read_text() == contents
    assert not paths[0].exists()


def test_failed_publish_leaves_no_partial_result_and_can_retry(pending, monkeypatch):
    protocol, output, paths, calls, _ = pending
    replace = campaign.os.replace

    def fail(*args):
        raise OSError("simulated write failure")

    monkeypatch.setattr(campaign.os, "replace", fail)
    with pytest.raises(OSError, match="simulated write failure"):
        campaign.evaluate_test(protocol, output)
    assert not any(p.exists() for p in paths)
    assert not list(output.rglob("*.tmp"))
    monkeypatch.setattr(campaign.os, "replace", replace)
    campaign.evaluate_test(protocol, output)
    assert calls == [1, 1, 2, 3]
    assert all(p.exists() for p in paths)


def test_atomic_serialization_failure_preserves_destination(tmp_path):
    path = tmp_path / "test.json"
    path.write_text("original")
    with pytest.raises(ValueError):
        campaign.write_json_atomic(path, [{"return": float("nan")}])
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]
