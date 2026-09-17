"""Explicit TD semantics shared by baseline tests and future fitted Q targets."""

from dataclasses import dataclass


def bootstrap_mask(terminated, truncated, *, bootstrap_on_truncation=True,
                   final_observation_available=True):
    if terminated:
        return 0.0
    if truncated:
        if not bootstrap_on_truncation:
            return 0.0
        if not final_observation_available:
            raise ValueError("Cannot bootstrap a truncation without its final observation")
    return 1.0


@dataclass(frozen=True)
class TDUpdate:
    old_q: float
    target: float
    new_q: float
    bootstrap: float


def q_learning_update(old_q, reward, next_value, *, eta_q, gamma, terminated,
                      truncated, bootstrap_on_truncation=True):
    """next_value is lazy so real terminal states are never queried."""
    if not 0 < eta_q <= 1 or not 0 <= gamma < 1:
        raise ValueError("Require 0 < eta_q <= 1 and 0 <= gamma < 1")
    mask = bootstrap_mask(terminated, truncated,
                          bootstrap_on_truncation=bootstrap_on_truncation)
    target = float(reward) + (gamma * float(next_value()) if mask else 0.0)
    return TDUpdate(float(old_q), target, float(old_q + eta_q * (target - old_q)), mask)


def frozen_q_targets(transitions, target_model, encoder, *, gamma, bootstrap_on_truncation=True):
    """Build once from a single frozen model; terminal observations are never read."""
    import numpy as np
    from .replay import concatenate_states

    if not 0 <= gamma < 1:
        raise ValueError('Require 0 <= gamma < 1')
    transitions = tuple(transitions)
    targets = np.array([t.reward for t in transitions], dtype=float)
    indices, states = [], []
    for i, transition in enumerate(transitions):
        mask = bootstrap_mask(transition.terminated, transition.truncated,
                              bootstrap_on_truncation=bootstrap_on_truncation,
                              final_observation_available=transition.next_state is not None)
        if mask and gamma:
            if transition.next_state is None:
                raise ValueError('Bootstrap requires the final next observation')
            indices.append(i)
            states.append(transition.next_state)
    if states:
        predictions = target_model.predict(concatenate_states(states, encoder))
        if predictions.ndim != 2 or len(predictions) != len(states) or not np.isfinite(predictions).all():
            raise ValueError('Invalid target predictions')
        targets[indices] += gamma * predictions.max(axis=1)
    if not np.isfinite(targets).all():
        raise ValueError('Non-finite Q targets')
    targets.flags.writeable = False
    return targets
