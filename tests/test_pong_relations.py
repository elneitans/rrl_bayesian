"""Pong v1 semantics, catalog and JSON relational memory without ALE/OCAtari."""

from dataclasses import replace
import itertools
import json

import numpy as np
import pytest

from bayesian_rrtl.encoding import RelationKey
from bayesian_rrtl.game_registry import get_game_spec
from bayesian_rrtl.games.pong import PongRelationsV1
from bayesian_rrtl.objects import DetectedObject, ObjectFrame


def frame(player=(20, 30, 4, 12), ball=(60, 80, 2, 4), enemy=(100, 130, 4, 12)):
    return ObjectFrame(tuple(DetectedObject(name, box is not None, box)
                             for name, box in (("player", player), ("ball", ball), ("enemy", enemy))))


def facts(state):
    return {(f.name, f.obj1, f.obj2): f.value for f in state}


def temporal(state):
    return {k: v for k, v in facts(state).items() if k[2] and k[2].endswith("t-1")}


def test_fixed_catalog_exact_keys_before_any_observation():
    relations = get_game_spec("Pong", "ocatari_ram_v1").relation_factory()
    encoder = relations.encoder()
    pairs = (("player_t", "ball_t"), ("player_t", "enemy_t"), ("ball_t", "enemy_t"),
             ("player_t", "player_t-1"), ("ball_t", "ball_t-1"), ("enemy_t", "enemy_t-1"))
    expected = {RelationKey("comparative", axis, a, b) for a, b in pairs for axis in ("x", "y")}
    expected |= {RelationKey("logical", "present", obj + time)
                 for obj in ("player", "ball", "enemy") for time in ("_t", "_t-1")}
    expected |= {RelationKey("logical", "incontact", a, b)
                 for a, b in (("player_t", "ball_t"), ("ball_t", "enemy_t"))}
    assert len(encoder.catalog) == 20 and set(encoder.catalog) == expected
    first = relations.transform(frame())
    second = relations.transform(frame())
    assert len(first) == 14 and len(second) == 20
    assert encoder.transform([second]).observed.all()
    relations.transform(frame(None, None, None))
    relations.reset()
    assert relations.encoder().to_dict() == encoder.to_dict()


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("difference,expected", [(-5, "less"), (-4.001, "less"), (-4, "same"),
                                                (0, "same"), (4, "same"), (4.001, "more"), (5, "more")])
def test_spatial_centers_tolerances_and_screen_direction(axis, difference, expected):
    # Different sizes: a top-left comparison would fail these boundary fixtures.
    player = [50.0, 50.0, 10.0, 12.0]
    ball = [54.0, 54.0, 2.0, 4.0]  # identical center (55, 56)
    player[axis] += difference
    state = facts(PongRelationsV1().transform(frame(tuple(player), tuple(ball))))
    assert state[(('x', 'y')[axis], 'player_t', 'ball_t')] == expected
    assert state[(('y', 'x')[axis], 'player_t', 'ball_t')] == 'same'


def test_all_ordered_pairs_in_both_axes():
    state = facts(PongRelationsV1().transform(frame()))
    for a, b in (("player", "ball"), ("player", "enemy"), ("ball", "enemy")):
        assert state[("x", a + "_t", b + "_t")] == "less"
        assert state[("y", a + "_t", b + "_t")] == "less"


@pytest.mark.parametrize("delta,expected", [(-0.125, "less"), (0, "same"), (0.125, "more")])
def test_temporal_zero_tolerance_for_every_slot(delta, expected):
    relations = PongRelationsV1()
    initial = frame()
    relations.transform(initial)
    moved = ObjectFrame(tuple(DetectedObject(o.identity, True,
                                            (o.bbox[0] + delta, o.bbox[1] - delta, *o.bbox[2:]))
                              for o in initial.objects))
    state = facts(relations.transform(moved))
    for name in ("player", "ball", "enemy"):
        assert state[("x", name + "_t", name + "_t-1")] == expected
        assert state[("y", name + "_t", name + "_t-1")] == {"less": "more", "same": "same", "more": "less"}[expected]


@pytest.mark.parametrize("ball,expected", [((1, 1, 2, 2), True), ((4, 1, 2, 2), True),
                                            ((4, 4, 2, 2), True), ((4.01, 1, 2, 2), False),
                                            ((1, 4.01, 2, 2), False), ((-2, -2, 2, 2), True)])
def test_contact_overlap_touch_corner_and_separation(ball, expected):
    state = facts(PongRelationsV1().transform(frame((0, 0, 4, 4), ball, (0, 0, 4, 4))))
    assert state[("incontact", "player_t", "ball_t")] is expected
    assert state[("incontact", "ball_t", "enemy_t")] is expected
    assert ("incontact", "player_t", "enemy_t") not in state


def test_first_frame_absence_is_explicit_and_encoding_distinguishes_missing():
    relations = PongRelationsV1()
    state = relations.transform(frame(ball=None))
    values = facts(state)
    assert values[("present", "player_t", None)] is True
    assert values[("present", "ball_t", None)] is False
    assert all(values[("present", name + "_t-1", None)] is False for name in ("player", "ball", "enemy"))
    assert len(state) == 8  # six presences plus paddle/paddle spatial relations
    assert values[("y", "player_t", "enemy_t")] == "less"
    encoder = relations.encoder()
    encoded = encoder.transform([state])
    presence = encoder.catalog.index(RelationKey("logical", "present", "ball_t"))
    spatial = encoder.catalog.index(RelationKey("comparative", "x", "player_t", "ball_t"))
    assert encoded.observed[0, presence] and encoded.values[0, presence] == 0
    assert not encoded.observed[0, spatial] and encoded.values[0, spatial] == -1


@pytest.mark.parametrize("missing", ["player", "ball", "enemy"])
def test_missing_slot_omits_only_dependent_facts_and_breaks_history(missing):
    relations = PongRelationsV1()
    relations.transform(frame())
    absent = frame(**{missing: None})
    state = relations.transform(absent)
    for f in state:
        if f.name != "present":
            assert missing + "_t" not in (f.obj1, f.obj2)
    reappeared = relations.transform(frame())
    assert facts(reappeared)[("present", missing + "_t-1", None)] is False
    assert all(key[1] != missing + "_t" for key in temporal(reappeared))
    assert len(temporal(reappeared)) == 4  # other objects retained their consecutive history
    assert len(temporal(relations.transform(frame()))) == 6


@pytest.mark.parametrize("reward", [-1, 1, 0.1])
def test_point_clears_all_history_before_state_then_saves_current(reward):
    relations = PongRelationsV1()
    relations.transform(frame())
    current = frame(player=(22, 31, 4, 12))
    point = relations.transform(current, raw_reward=reward)
    assert not temporal(point)
    assert all(f.value is False for f in point if f.name == "present" and f.obj1.endswith("t-1"))
    assert len(point) == 14
    following = relations.transform(current, raw_reward=0)
    assert len(temporal(following)) == 6 and set(temporal(following).values()) == {"same"}
    assert relations.previous == current


def test_reset_and_all_absent_frame():
    relations = PongRelationsV1()
    relations.transform(frame())
    relations.reset()
    reset_state = relations.transform(frame())
    assert reset_state == PongRelationsV1().transform(frame())
    relations.reset()
    state = relations.transform(frame(None, None, None))
    assert len(state) == 6 and all(f.value is False for f in state)


def test_objects_and_facts_order_do_not_change_encoding():
    source = frame()
    relations = PongRelationsV1()
    expected = relations.encoder().transform([relations.transform(source)])
    for objects in itertools.permutations(source.objects):
        state = PongRelationsV1().transform(ObjectFrame(objects))
        for ordered in (list(state), list(reversed(list(state)))):
            actual = relations.encoder().transform([ordered])
            np.testing.assert_array_equal(actual.values, expected.values)
            np.testing.assert_array_equal(actual.observed, expected.observed)
            assert actual.catalog_hash == expected.catalog_hash


@pytest.mark.parametrize("history", [None, frame(), frame(ball=None)])
def test_json_snapshot_roundtrip_preserves_future_facts(history):
    left, right = PongRelationsV1(), PongRelationsV1()
    if history is not None:
        left.transform(history)
    saved = json.loads(json.dumps(left.snapshot()))
    right.restore(saved)
    # Mutating the serialized input cannot mutate the restored memory.
    if saved["previous"] is not None:
        saved["previous"]["objects"][0]["bbox"] = [900, 900, 1, 1]
    for next_frame, reward in ((frame(), 0), (frame(ball=None), 1), (frame(), 0), (frame(), 0)):
        assert left.transform(next_frame, raw_reward=reward) == right.transform(next_frame, raw_reward=reward)
        assert left.snapshot() == right.snapshot()


@pytest.mark.parametrize("change", ["schema", "config", "hash", "kind", "frame"])
def test_incompatible_snapshot_rejected_without_changing_memory(change):
    relations = PongRelationsV1()
    relations.transform(frame())
    before = relations.snapshot()
    bad = json.loads(json.dumps(before))
    if change == "schema":
        bad["manifest"]["relation_schema"] = "pong_relational_v2"
    elif change == "config":
        bad["manifest"]["config"]["spatial_tolerance"] = 5  # unchanged catalog hash is insufficient
    elif change == "hash":
        bad["manifest"]["catalog_hash"] = "other"
    elif change == "kind":
        bad["kind"] = "different"
    else:
        bad["previous"]["width"] = 320
    with pytest.raises(ValueError, match="snapshot"):
        relations.restore(bad)
    assert relations.snapshot() == before


@pytest.mark.parametrize("kwargs", [{"representation": "logical"}, {"include_incomplete_states": False},
                                    {"include_incomplete_states": 1}])
def test_unsupported_relation_configuration(kwargs):
    with pytest.raises(ValueError, match="requires"):
        PongRelationsV1(**kwargs)


def test_invalid_frame_and_reward_do_not_change_memory():
    relations = PongRelationsV1()
    original = frame()
    relations.transform(original)
    for invalid in (replace(original, width=320), ObjectFrame(original.objects[:2])):
        with pytest.raises(ValueError):
            relations.transform(invalid)
        assert relations.previous == original
    with pytest.raises(ValueError, match="finite"):
        relations.transform(original, raw_reward=float("nan"))
    assert relations.previous == original


def test_new_builder_does_not_call_legacy_relations(monkeypatch):
    import preprocessing
    monkeypatch.setattr(preprocessing, "get_state_comparative_pong",
                        lambda *args: pytest.fail("legacy relation builder called"))
    assert PongRelationsV1().transform(frame())
