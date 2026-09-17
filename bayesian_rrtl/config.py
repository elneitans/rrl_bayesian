"""Complete JSON configuration for the corrected baseline; no environment imports."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class BaselineConfig:
    variant: str = "rrtl_corrected"
    game: str = "Breakout"
    model_version: str = "comparative"
    run: int = 1
    seed: int = 1
    n_iterations: int = 256
    results_dir: str = "results"
    env_id: str | None = None
    eta_q: float = 0.025
    gamma: float = 0.99
    epsilon_init: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay_steps: int = 500000
    max_depth: int = 10
    min_sample_size: int = 100000
    significance_level: float = 0.0001
    action_buffer_capacity: int = 10
    best_literal_criteria: str = "p-value"
    include_incomplete_states: bool = False
    inherit_q_values: bool = True
    resample_repeated_actions: bool = True
    bootstrap_on_truncation: bool = True
    frameskip: int = 4
    repeat_action_probability: float = 0.0
    max_episode_steps: int = 27000
    validation_seeds: tuple[int, ...] = (100001, 100002)
    test_seeds: tuple[int, ...] = (200001, 200002)
    render_graph: bool = False

    def __post_init__(self):
        object.__setattr__(self, "validation_seeds", tuple(self.validation_seeds))
        object.__setattr__(self, "test_seeds", tuple(self.test_seeds))
        if self.variant != "rrtl_corrected":
            raise ValueError("Use train_and_test.py for the unmodified rrtl_legacy")
        if self.game not in ("Breakout", "Pong", "DemonAttack"):
            raise ValueError("Unsupported game")
        if self.model_version not in ("comparative", "logical"):
            raise ValueError("Unsupported relational representation")
        if not 0 < self.eta_q <= 1 or not 0 <= self.gamma < 1:
            raise ValueError("Invalid learning rate or discount")
        if not 0 < self.epsilon_min <= self.epsilon_init <= 1:
            raise ValueError("Require 0 < epsilon_min <= epsilon_init <= 1")
        for value in (self.n_iterations, self.epsilon_decay_steps, self.min_sample_size,
                      self.action_buffer_capacity, self.frameskip, self.max_episode_steps, self.run):
            if not isinstance(value, int) or value < 1:
                raise ValueError("Iteration counts, run and capacities must be positive integers")
        if self.max_depth < 0 or not 0 < self.significance_level < 1:
            raise ValueError("Invalid depth or significance level")
        if not 0 <= self.repeat_action_probability <= 1:
            raise ValueError("Invalid repeat action probability")
        if self.best_literal_criteria not in ("p-value", "f-ratio", "max-q-value"):
            raise ValueError("Invalid literal criterion")
        seeds = [self.seed, *self.validation_seeds, *self.test_seeds]
        if any(not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
            raise ValueError("Seeds must be unsigned 32-bit integers")
        if len(seeds) != len(set(seeds)):
            raise ValueError("Training, validation and test seed lists must be disjoint")

    @property
    def resolved_env_id(self):
        return self.env_id or f"{self.game}NoFrameskip-v4"

    def to_dict(self):
        return asdict(self)

    @property
    def config_hash(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
