from collections import namedtuple
from dataclasses import FrozenInstanceError
import itertools
import math

import numpy as np
import pytest

from bayesian_rrtl import RelationKey, RelationalEncoder, Rule, RuleCatalog, Tree

Fact = namedtuple("Fact", "type value name obj1 obj2 obj3", defaults=[None, None])
X = RelationKey("comparative", "x", "player_t", "ball_t")
PRESENT = RelationKey("logical", "present", "ball_t")


def fact(key, value):
    return Fact(key.type, value, key.name, key.obj1, key.obj2, key.obj3)


def dataset():
    return [[fact(X, "less"), fact(PRESENT, "True")],
            [fact(X, "same"), fact(PRESENT, "False")],
            [fact(X, "more")], []]


def test_fact_and_catalog_order_do_not_affect_encoding():
    keys = [X, PRESENT, RelationKey("comparative", "y", "ball_t", "ball_t-1"),
            RelationKey("logical", "between", "a_t", "b_t", "c_t")]
    facts = [fact(k, "same" if k.type == "comparative" else "False") for k in keys]
    expected = RelationalEncoder(keys).transform([facts])
    for permutation in itertools.permutations(facts):
        encoder = RelationalEncoder.from_states([permutation])
        batch = encoder.transform([permutation])
        np.testing.assert_array_equal(batch.values, expected.values)
        assert batch.catalog_hash == expected.catalog_hash
    assert len(RelationalEncoder(keys + [RelationKey("comparative", "x", "player_t-1", "ball_t")]).catalog) == 5


def test_false_missing_categories_and_empty_state():
    encoder = RelationalEncoder([X, PRESENT])
    batch = encoder.transform(dataset())
    np.testing.assert_array_equal(Rule(PRESENT, "equals", "False").route(batch), [0, 1, 0, 0])
    np.testing.assert_array_equal(Rule(PRESENT, "is_missing").route(batch), [0, 0, 1, 1])
    np.testing.assert_array_equal(Rule(X, "equals", "more").route(batch), [0, 0, 1, 0])
    assert not batch.observed[-1].any()
    assert (batch.values[-1] == -1).all()
    with pytest.raises(ValueError):
        batch.values[0, 0] = 2


def test_encoder_round_trip_and_invalid_data():
    encoder = RelationalEncoder([X, PRESENT])
    restored = RelationalEncoder.from_dict(encoder.to_dict())
    np.testing.assert_array_equal(restored.transform(dataset()).values, encoder.transform(dataset()).values)
    with pytest.raises(ValueError, match="Conflicting"):
        encoder.transform([[fact(X, "less"), fact(X, "more")]])
    with pytest.raises(ValueError, match="Unknown"):
        RelationalEncoder([PRESENT]).transform([[fact(X, "less")]])
    with pytest.raises(ValueError, match="Invalid value"):
        encoder.transform([[fact(PRESENT, "missing")]])
    corrupted = encoder.to_dict() | {"catalog_hash": "wrong"}
    with pytest.raises(ValueError, match="hash"):
        RelationalEncoder.from_dict(corrupted)


def test_total_routing_and_no_internal_fallback():
    batch = RelationalEncoder([X, PRESENT]).transform(dataset())
    tree = Tree(mu=999, rule=Rule(X, "equals", "more"), true_child=Tree(7),
                false_child=Tree(rule=Rule(PRESENT, "is_missing"),
                                 true_child=Tree(-3), false_child=Tree(2)))
    np.testing.assert_array_equal(tree.predict(batch), [2, 2, 7, -3])
    indices = np.concatenate([rows for leaf, rows in tree.partition(batch).values()])
    np.testing.assert_array_equal(np.sort(indices), np.arange(len(batch)))
    assert tree.predict(batch.take([])).shape == (0,)
    with pytest.raises(FrozenInstanceError):
        tree.mu = 4
    with pytest.raises(ValueError):
        Tree(rule=Rule(X, "equals", "more"), true_child=Tree(1))


def test_rule_eligibility_hierarchical_probabilities_and_support():
    encoder = RelationalEncoder([X, PRESENT])
    batch = encoder.transform(dataset())
    rules = RuleCatalog(encoder)
    groups = rules.eligible(batch)
    masses = [sum(math.exp(rules.log_probability(r, groups)) for r in group)
              for group in groups.values()]
    np.testing.assert_allclose(masses, [0.5, 0.5])
    assert rules.eligible(batch, depth=6) == {}
    assert rules.eligible(batch, [0]) == {}
    assert rules.eligible(batch, min_child=3) == {}
    split = Rule(X, "equals", "more")
    valid = Tree(rule=split, true_child=Tree(1), false_child=Tree(0))
    valid.validate_support(batch, rules)
    invalid = Tree(rule=split, true_child=valid, false_child=Tree(0))
    with pytest.raises(ValueError, match="support"):
        invalid.validate_support(batch, rules)
    with pytest.raises(ValueError, match="support"):
        valid.validate_support(batch, rules, max_depth=0)
    with pytest.raises(ValueError, match="differ"):
        rules.eligible(RelationalEncoder([PRESENT]).transform([[]]))


def test_empty_catalog_root_prediction():
    encoder = RelationalEncoder([])
    batch = encoder.transform([[], []])
    np.testing.assert_array_equal(Tree(3).predict(batch), [3, 3])
    assert RuleCatalog(encoder).eligible(batch) == {}
