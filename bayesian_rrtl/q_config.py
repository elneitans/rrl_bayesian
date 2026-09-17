"""Configuration for block fitted Q; no Atari imports or incremental TD rate."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from .priors import FixedScale, TreePrior, leaf_variance


@dataclass(frozen=True)
class BayesianQConfig:
    actions: tuple[str, ...] = ('advance', 'exit')
    seed: int = 1
    gamma: float = 0.8
    reward_min: float = -0.2
    reward_max: float = 1.0
    replay_capacity: int = 2000
    batch_size_per_action: int = 128
    fit_interval: int = 100
    warmup_interactions: int = 0
    kernel: str = "grow_prune"
    m: int = 5
    burn_in: int = 100
    draws: int = 100
    k: float = 2.0
    alpha_tree: float = 0.95
    beta_tree: float = 2.0
    max_depth: int = 4
    min_child: int = 1
    nu: float = 3.0
    noise_quantile: float = 0.9
    variance_floor: float = 1e-8
    epsilon_init: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay_steps: int = 1000
    bootstrap_on_truncation: bool = True
    resample_repeated_actions: bool = False
    action_buffer_capacity: int = 10
    validation_seeds: tuple[int, ...] = (100001, 100002)
    test_seeds: tuple[int, ...] = (200001, 200002)

    def __post_init__(self):
        for name in ('actions', 'validation_seeds', 'test_seeds'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.actions or len(set(self.actions)) != len(self.actions) or any(
                not isinstance(a, str) or not a for a in self.actions):
            raise ValueError('Actions must be nonempty unique names in a fixed order')
        for name in ('replay_capacity', 'batch_size_per_action', 'fit_interval', 'm', 'draws',
                     'epsilon_decay_steps', 'action_buffer_capacity'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.burn_in) is not int or self.burn_in < 0:
            raise ValueError('burn_in must be a nonnegative integer')
        if type(self.warmup_interactions) is not int or self.warmup_interactions < 0:
            raise ValueError('warmup_interactions must be a nonnegative integer')
        if not 0 < self.epsilon_min <= self.epsilon_init <= 1:
            raise ValueError('Require 0 < epsilon_min <= epsilon_init <= 1')
        if not math.isfinite(self.nu) or self.nu <= 0 or not 0 < self.noise_quantile < 1:
            raise ValueError('Invalid noise calibration')
        if not math.isfinite(self.variance_floor) or self.variance_floor <= 0:
            raise ValueError('variance_floor must be positive and finite')
        for name in ('bootstrap_on_truncation', 'resample_repeated_actions'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        seeds = (self.seed, *self.validation_seeds, *self.test_seeds)
        if any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError('Training, validation and test seeds must be disjoint uint32 values')
        if self.kernel not in ("grow_prune", "revision"):
            raise ValueError("Unknown kernel")
        self.scale
        self.tree_prior
        leaf_variance(self.k, self.m)

    @property
    def scale(self):
        return FixedScale.from_rewards(self.reward_min, self.reward_max, self.gamma)

    @property
    def tree_prior(self):
        return TreePrior(self.alpha_tree, self.beta_tree, self.max_depth, self.min_child)

    def to_dict(self):
        return asdict(self)

    @property
    def config_hash(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
