"""Environment-free TD/F-test adapter; delegates learning to corrected H0.

Only this boundary normalizes boolean facts. Missing literals retain Node's
internal-node Q fallback; no missing branch or new splitting formula is added.
Episode collection, scheduled exploration and checkpoint files belong to the
subsequent comparison-plan stages.
"""

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np

from .baseline import CorrectedNode, CorrectedRRLAgent
from .encoding import CATEGORIES, RelationKey, RelationalEncoder
from .targets import bootstrap_mask
from preprocessing import Fact
from relational_regresion_tree import Literal

LEARNER_ID = 'rrtl_corrected_common_v1'


@dataclass(frozen=True)
class BaselineLearnerConfig:
    eta_q: float = .025
    gamma: float = .99
    max_depth: int = 4
    min_sample_size: int = 128
    significance_level: float = .0001
    best_literal_criteria: str = 'p-value'
    inherit_q_values: bool = True
    bootstrap_on_truncation: bool = True
    seed: int = 1

    def __post_init__(self):
        if not 0 < self.eta_q <= 1 or not 0 <= self.gamma < 1:
            raise ValueError('Invalid learning rate or discount')
        for name, minimum in (('max_depth', 0), ('min_sample_size', 1), ('seed', 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'Invalid {name}')
        if self.seed >= 2**32 or not 0 < self.significance_level < 1:
            raise ValueError('Invalid seed or significance level')
        if self.best_literal_criteria not in ('p-value', 'f-ratio', 'max-q-value'):
            raise ValueError('Invalid literal criterion')
        if any(type(getattr(self, name)) is not bool for name in
               ('inherit_q_values', 'bootstrap_on_truncation')):
            raise ValueError('Inheritance and bootstrap options must be boolean')


def normalize_facts(state):
    """Return deterministic immutable facts; never turn absence into False."""
    values = {}
    for fact in state:
        key = RelationKey.from_fact(fact)
        value = str(bool(fact.value)) if isinstance(fact.value, (bool, np.bool_)) else fact.value
        if value not in CATEGORIES[key.type]:
            raise ValueError(f'Invalid value for {key}: {value!r}')
        if key in values and values[key] != value:
            raise ValueError(f'Conflicting facts for {key}')
        values[key] = value
    return tuple(Fact(key.type, values[key], key.name, key.obj1, key.obj2, key.obj3)
                 for key in sorted(values, key=RelationKey.sort_key))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class RelationalBaselineAgent:
    learner_id = LEARNER_ID

    def __init__(self, actions, encoder, config=BaselineLearnerConfig(), *, ordered_catalog=None):
        self.actions = tuple(actions)
        if not self.actions or len(set(self.actions)) != len(self.actions) or any(
                not isinstance(a, str) or not a for a in self.actions):
            raise ValueError('Actions must be unique names in environment order')
        self.config = config
        # Own the encoder: callers cannot change the hypothesis support after setup.
        self.encoder = RelationalEncoder.from_dict(encoder.to_dict())
        catalog = tuple(self.encoder.catalog if ordered_catalog is None else ordered_catalog)
        if len(set(catalog)) != len(catalog) or not set(catalog) <= set(self.encoder.catalog):
            raise ValueError('Ordered candidate catalog must be a unique subset of the input encoder')
        self.catalog = catalog
        self.catalog_source = 'environment_encoder' if ordered_catalog is None else 'explicit_historical_order'
        self.literals = tuple(Literal(k.name, k.type, k.obj1, k.obj2, k.obj3) for k in catalog)
        # Node matches name/obj1/obj2 only; reject ambiguous contracts rather than
        # pretending obj3 or type can disambiguate a historical lookup.
        lookup = [(k.name, k.obj1, k.obj2) for k in self.encoder.catalog]
        if len(set(lookup)) != len(lookup):
            raise ValueError('Input catalog is ambiguous for historical Node lookup')
        roots = {action: CorrectedNode(root_action=action, refinements=list(self.literals),
                 max_depth=config.max_depth, min_sample_size=config.min_sample_size,
                 significance_level=config.significance_level,
                 best_literal_criteria=config.best_literal_criteria, valid_actions=list(self.actions))
                 for action in self.actions}
        # Controlled receiver for the existing update method: no RRLAgent
        # constructor, image preprocessor, Gym instance or output directory.
        self._learner = SimpleNamespace(config=config, gamma=config.gamma,
            inherit_q_values=config.inherit_q_values, root_nodes=roots,
            index_to_action=dict(enumerate(self.actions)), n_splits=0)
        self.interactions = 0
        self.policy_rng = np.random.default_rng(config.seed)
        self.query_counts = {purpose: {'queries': 0, 'internal_fallbacks': 0}
                             for purpose in ('prediction', 'update', 'bootstrap')}

    @classmethod
    def from_environment(cls, environment, config=BaselineLearnerConfig(), *, ordered_catalog=None):
        """Consume the actual adapter's action/encoder contracts, no new emulator."""
        if tuple(environment.actions) != environment.spec.actions:
            raise ValueError('Actual environment actions differ from registered specification')
        return cls(environment.actions, environment.encoder(), config, ordered_catalog=ordered_catalog)

    @property
    def root_nodes(self):
        return self._learner.root_nodes

    @property
    def n_splits(self):
        return self._learner.n_splits

    def _state(self, state):
        normalized = normalize_facts(state)
        # Validate values and known input relations; never grow the fixed catalog.
        self.encoder.transform([normalized])
        return normalized

    def _record_queries(self, state, roots, purpose):
        """Count semantic per-action queries, not Node's repeated traversal calls."""
        counts = self.query_counts[purpose]
        for root in roots:
            counts['queries'] += 1
            counts['internal_fallbacks'] += int(root.get_state_leaf(state).literal is not None)

    def predict_q(self, state):
        normalized = self._state(state)
        roots = [self.root_nodes[action] for action in self.actions]
        self._record_queries(normalized, roots, 'prediction')
        return np.array([root.predict(normalized) for root in roots], dtype=float)

    def select_action(self, state, *, epsilon=0., rng=None):
        """Caller supplies epsilon; uniform greedy ties, no repeated-action buffer.

        The scheduled policy/episode RNG contract is implemented by plan B.
        """
        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError('epsilon must be in [0,1]')
        normalized = self._state(state)
        rng = self.policy_rng if rng is None else rng
        if epsilon and rng.random() < epsilon:
            return int(rng.integers(len(self.actions)))
        q = self.predict_q(normalized)
        return int(rng.choice(np.flatnonzero(q == q.max())))

    def observe(self, state, action, reward, next_state, terminated, truncated):
        """One transformed reward, one TD update, then historical statistics/split."""
        if type(action) is not int or not 0 <= action < len(self.actions):
            raise ValueError('Action index outside environment catalog')
        if not math.isfinite(reward) or type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError('Need finite reward and boolean flags')
        normalized = self._state(state)
        bootstrap = bootstrap_mask(terminated, truncated,
            bootstrap_on_truncation=self.config.bootstrap_on_truncation,
            final_observation_available=next_state is not None)
        if bootstrap and next_state is None:
            raise ValueError('Bootstrap requires next state')
        following = self._state(next_state) if bootstrap else ()
        self._record_queries(normalized, [self.root_nodes[self.actions[action]]], 'update')
        if bootstrap:
            self._record_queries(following, self.root_nodes.values(), 'bootstrap')
        update = CorrectedRRLAgent.update_transition(
            self._learner, normalized, action, float(reward), following, terminated, truncated)
        self.interactions += 1
        return update

    def contract(self):
        literals = [dict(literal._asdict()) for literal in self.literals]
        result = {'learner': self.learner_id, 'actions': list(self.actions),
                  'config': asdict(self.config), 'encoder': self.encoder.to_dict(),
                  'catalog_source': self.catalog_source,
                  'ordered_catalog': [asdict(key) for key in self.catalog],
                  'ordered_literals': literals, 'ordered_literals_hash': _hash(literals),
                  'fact_values': 'bool_to_strings_v1',
                  'missing_literal': 'historical_internal_node_q',
                  'selection': 'explicit_epsilon_uniform_greedy_ties_v1'}
        return {**result, 'contract_hash': _hash(result)}

    def state_dict(self):
        """Detached pickle-compatible learner data, no environment/collector.

        Atomic on-disk checkpoint format and collector validation belong to C.
        """
        return copy.deepcopy({'version': 1, 'contract': self.contract(),
            'roots': self.root_nodes, 'interactions': self.interactions,
            'n_splits': self.n_splits, 'policy_rng': self.policy_rng.bit_generator.state,
            'query_counts': self.query_counts})

    def load_state_dict(self, state):
        required = {'version', 'contract', 'roots', 'interactions', 'n_splits', 'policy_rng', 'query_counts'}
        if set(state) != required or state['version'] != 1 or state['contract'] != self.contract():
            raise ValueError('Baseline learner state/contract mismatch')
        if list(state['roots']) != list(self.actions) or any(
                type(state[key]) is not int or state[key] < 0 for key in ('interactions', 'n_splits')):
            raise ValueError('Invalid learner roots/counters')
        counts = state['query_counts']
        if not isinstance(counts, dict) or set(counts) != {'prediction', 'update', 'bootstrap'}:
            raise ValueError('Invalid query counters')
        for count in counts.values():
            if (not isinstance(count, dict) or set(count) != {'queries', 'internal_fallbacks'}
                    or any(type(value) is not int or value < 0 for value in count.values())
                    or count['internal_fallbacks'] > count['queries']):
                raise ValueError('Invalid query counters')
        saved = copy.deepcopy(state)
        rng = np.random.default_rng()
        rng.bit_generator.state = saved['policy_rng']
        self._learner.root_nodes = saved['roots']
        self._learner.n_splits = saved['n_splits']
        self.interactions = saved['interactions']
        self.policy_rng = rng
        self.query_counts = saved['query_counts']

    def predict_posterior(self, *args, **kwargs):
        raise NotImplementedError('TD/F-test baseline has no posterior or draws')

    def predict_draws(self, *args, **kwargs):
        raise NotImplementedError('TD/F-test baseline has no posterior or draws')
