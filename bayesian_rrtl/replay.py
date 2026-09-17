"""Finite FIFO replay, immutable encoded transitions and monotonic IDs."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .encoding import EncodedBatch


def concatenate_states(states, encoder):
    states = tuple(states)
    if not states:
        return encoder.transform([])
    if any(len(s) != 1 or s.catalog_hash != encoder.catalog_hash for s in states):
        raise ValueError('States must be single rows from the fixed encoder')
    return EncodedBatch(np.concatenate([s.values for s in states]),
                        np.concatenate([s.observed for s in states]),
                        encoder.catalog, encoder.catalog_hash)


@dataclass(frozen=True)
class Transition:
    transition_id: int
    state: EncodedBatch
    action: int
    reward: float
    raw_reward: float
    next_state: EncodedBatch | None
    terminated: bool
    truncated: bool
    episode_id: int
    step_id: int

    def __post_init__(self):
        for name in ('transition_id', 'action', 'episode_id', 'step_id'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if not math.isfinite(self.reward) or not math.isfinite(self.raw_reward):
            raise ValueError('Rewards must be finite')
        if len(self.state) != 1:
            raise ValueError('A transition needs exactly one state')
        if self.next_state is not None and (len(self.next_state) != 1 or
                                            self.state.catalog_hash != self.next_state.catalog_hash):
            raise ValueError('Next state must have the same catalog')
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError('Terminal flags must be boolean')


class ReplayBuffer:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Replay capacity must be a positive integer')
        self.capacity = capacity
        self.next_id = 0
        self._items = deque(maxlen=capacity)

    def __len__(self):
        return len(self._items)

    @property
    def transitions(self):
        return tuple(self._items)

    def append(self, **fields):
        transition = Transition(transition_id=self.next_id, **fields)
        self._items.append(transition)
        self.next_id += 1
        return transition

    def sample(self, action, size, rng):
        if type(size) is not int or size < 1:
            raise ValueError('Sample size must be a positive integer')
        candidates = [t for t in self._items if t.action == action]
        indices = np.sort(rng.choice(len(candidates), min(size, len(candidates)), replace=False))
        return tuple(candidates[i] for i in indices)
