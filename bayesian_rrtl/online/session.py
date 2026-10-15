"""Bounded online collector, disjoint RNG streams and phase accounting."""
import copy
from dataclasses import asdict
import time

import numpy as np

from bayesian_rrtl.encoding import RelationalEncoder
from bayesian_rrtl.online.learners import LEARNERS, LearningConfig, learner_metrics, make_agent


class TimedEncoder(RelationalEncoder):
    def __init__(self, catalog):
        super().__init__(catalog)
        self.seconds = 0.

    def transform(self, states):
        started = time.perf_counter()
        result = super().transform(states)
        self.seconds += time.perf_counter()-started
        return result


class OnlineSession:
    schema = 'pong_online_session_v2'

    def __init__(self, cell, environment, *, cache_bytes=1048576):
        self.cell = copy.deepcopy(cell)
        self.learning = LearningConfig(**cell['learning'])
        self.learner = cell['learner']
        self.agent = make_agent(self.learner, self.learning, environment.encoder(), cell['estimator'],
                                cache_bytes=cache_bytes)
        if tuple(environment.actions) != self.learning.actions:
            raise ValueError('Action contract mismatch')
        if not LEARNERS[self.learner].batch:
            self.agent.encoder = TimedEncoder(self.agent.encoder.catalog)
        self.environment = environment
        a, b = np.random.SeedSequence(self.learning.seed).spawn(2)
        self.environment_rng, self.exploration_rng = np.random.default_rng(a), np.random.default_rng(b)
        if LEARNERS[self.learner].batch:
            self.agent.environment_rng = self.environment_rng
            self.agent.exploration_rng = self.exploration_rng
        self.state = None
        self.done = True
        self.episode_id = -1
        self.step_id = 0
        self.episode_seed = None
        self.raw_return = self.points_won = self.points_lost = 0.
        self.pending_rows = []
        self.checkpoint_ready = True
        self.phase_seconds = {'emulation_extraction': 0., 'relations_encoding': 0.,
                              'prediction': 0., 'policy_collection_td': 0.,
                              'selection_targets': 0., 'fit': 0., 'checkpoint_logs': 0.}
        self.ended_episodes = self.completed_games = self.truncated_episodes = 0
        self.last_fit_cost = 0.

    @property
    def interactions(self):
        return self.agent.interactions

    @property
    def epsilon(self):
        c = self.learning
        return max(c.epsilon_min, c.epsilon_init*(c.epsilon_min/c.epsilon_init)**(
            self.interactions/c.epsilon_decay_steps))

    def next_fit_due(self):
        if not LEARNERS[self.learner].batch:
            return False
        n = self.interactions+1
        return n >= self.learning.warmup_interactions and n-self.agent.last_fit_step >= self.learning.fit_interval

    def _reset(self):
        reserved = set(self.learning.validation_seeds+self.learning.test_seeds)
        while True:
            seed = int(self.environment_rng.integers(2**32))
            if seed not in reserved:
                break
        self.episode_seed = seed
        self.state = self.environment.reset(seed=seed)
        self.episode_id += 1
        self.step_id = 0
        self.raw_return = self.points_won = self.points_lost = 0.
        self.done = False

    def step(self):
        if not self.checkpoint_ready:
            raise ValueError('Incomplete update; restore durable checkpoint')
        self.checkpoint_ready = False
        if self.done:
            start = time.perf_counter()
            self._reset()
            reset_seconds = time.perf_counter()-start
            relation_seconds = getattr(self.environment, 'last_relation_seconds', 0.)
            self.phase_seconds['emulation_extraction'] += max(0., reset_seconds-relation_seconds)
            self.phase_seconds['relations_encoding'] += relation_seconds
        started = time.perf_counter()
        epsilon = self.epsilon
        q_elapsed = 0.
        rng = self.exploration_rng
        phase_before = dict(self.agent.phase_seconds) if LEARNERS[self.learner].batch else None
        baseline_encoding = self.agent.encoder.seconds if phase_before is None else 0.
        if rng.random() < epsilon:
            action = int(rng.integers(len(self.learning.actions)))
        elif LEARNERS[self.learner].batch:
            action = self.agent.select_action(self.state, explore=False)
        else:
            q_start = time.perf_counter()
            q = self.agent.predict_q(self.state)
            q_elapsed = time.perf_counter()-q_start
            q_encoding = self.agent.encoder.seconds-baseline_encoding
            self.phase_seconds['prediction'] += max(0., q_elapsed-q_encoding)
            action = int(rng.choice(np.flatnonzero(q == q.max())))
        action_seconds = time.perf_counter()-started
        start = time.perf_counter()
        following, reward, raw, terminal, truncated = self.environment.step(action)
        env_seconds = time.perf_counter()-start
        relation_seconds = getattr(self.environment, 'last_relation_seconds', 0.)
        self.phase_seconds['emulation_extraction'] += max(0., env_seconds-relation_seconds)
        self.phase_seconds['relations_encoding'] += relation_seconds
        start = time.perf_counter()
        detail = None
        if LEARNERS[self.learner].batch:
            self.agent.observe(self.state, action, reward, following, terminal, truncated,
                raw_reward=raw, episode_id=self.episode_id, step_id=self.step_id)
        else:
            detail = asdict(self.agent.observe(self.state, action, reward, following, terminal, truncated))
        observe_seconds = time.perf_counter()-start
        fitted = None
        if LEARNERS[self.learner].batch:
            fitted = self.agent.fit_if_due()
            for phase, destination in [('selection_targets', 'selection_targets'), ('fit', 'fit'),
                                       ('prediction', 'prediction'), ('encoding', 'relations_encoding')]:
                self.phase_seconds[destination] += self.agent.phase_seconds[phase]-phase_before[phase]
            prediction_encoding = sum(self.agent.phase_seconds[k]-phase_before[k] for k in ('prediction', 'encoding'))
            self.phase_seconds['policy_collection_td'] += max(0., action_seconds+observe_seconds-prediction_encoding)
            self.last_fit_cost = max(self.last_fit_cost, self.agent.last_fit_seconds)
        else:
            encoding = self.agent.encoder.seconds-baseline_encoding
            self.phase_seconds['relations_encoding'] += encoding
            self.phase_seconds['policy_collection_td'] += max(0., action_seconds+observe_seconds-q_elapsed-
                (encoding-q_encoding if q_elapsed else encoding))
        self.state, self.done = following, bool(terminal or truncated)
        self.raw_return += raw
        self.points_won += max(float(raw), 0.)
        self.points_lost += max(-float(raw), 0.)
        row = {'interaction': self.interactions, 'transition_id': self.interactions-1,
               'episode_id': self.episode_id, 'step_id': self.step_id, 'episode_seed': self.episode_seed,
               'epsilon': epsilon, 'action': action, 'reward': float(reward), 'raw_reward': float(raw),
               'terminated': bool(terminal), 'truncated': bool(truncated), 'episode_end': self.done,
               'raw_episode_return': self.raw_return, 'points_won': self.points_won,
               'points_lost': self.points_lost, 'fit_id': self.agent.model.model_id if LEARNERS[self.learner].batch else None,
               'fit': None if fitted is None else {'target_model_id': fitted.target_model_id,
                       'actions': fitted.diagnostics}, 'td': detail}
        self.step_id += 1
        self.ended_episodes += int(self.done)
        self.completed_games += int(terminal)
        self.truncated_episodes += int(truncated and not terminal)
        if self.cell.get('game','Pong') != 'Pong':
            row.pop('points_won',None)
            row.pop('points_lost',None)
        self.pending_rows.append(row)
        self.checkpoint_ready = True
        return row

    def metrics(self):
        return {'interactions': self.interactions, 'epsilon': self.epsilon,
                'relations': len(self.agent.encoder.catalog),
                'ended_episodes': self.ended_episodes, 'completed_games': self.completed_games,
                'truncated_episodes': self.truncated_episodes, 'phase_seconds': dict(self.phase_seconds),
                'learner': learner_metrics(self.agent, self.learner,
                    self.agent.encoder.transform([self.state]) if self.state is not None else self.agent.encoder.transform([]))}

    def state_dict(self):
        if not self.checkpoint_ready:
            raise ValueError('Checkpoint forbidden during incomplete update')
        state = {k: v for k, v in vars(self).items() if k not in ('environment', 'pending_rows')}
        return {'schema': self.schema, 'agent_type': type(self.agent).__name__,
                'collector': state, 'environment': self.environment.snapshot(),
                'runtime_environment': self.environment.runtime_manifest()}

    @classmethod
    def restore(cls, payload, cell, environment):
        if payload['schema'] != cls.schema or payload['collector']['cell'] != cell:
            raise ValueError('Session schema/cell mismatch')
        expected = 'OnlineBatchAgent' if LEARNERS[cell['learner']].batch else 'RelationalBaselineAgent'
        if payload['agent_type'] != expected or type(payload['collector']['agent']).__name__ != expected:
            raise ValueError('Exact checkpoint agent type mismatch')
        if payload['runtime_environment'] != environment.runtime_manifest():
            raise ValueError('Environment runtime mismatch')
        session = cls.__new__(cls)
        vars(session).update(payload['collector'])
        session.environment = environment
        session.pending_rows = []
        if session.agent.encoder.to_dict() != environment.encoder().to_dict():
            raise ValueError('Collector/environment encoder mismatch')
        if not session.checkpoint_ready:
            raise ValueError('Unfinished update checkpoint')
        if asdict(session.learning) != asdict(LearningConfig(**cell['learning'])):
            raise ValueError('Collector learning configuration mismatch')
        if LEARNERS[cell['learner']].batch:
            if (session.agent.learner_id != cell['learner'] or
                    session.agent.estimator_config != cell['estimator'] or
                    asdict(session.agent.config) != asdict(session.learning) or
                    session.agent.replay.next_id != session.interactions):
                raise ValueError('Batch checkpoint learner/config/replay mismatch')
            for model in (session.agent.model, session.agent.target_model):
                if model.catalog_hash != session.agent.encoder.catalog_hash or model.actions != session.learning.actions:
                    raise ValueError('Checkpoint model contract mismatch')
            for transition in session.agent.replay.transitions:
                for batch in (transition.state, transition.next_state):
                    if batch is not None:
                        batch.values.flags.writeable = batch.observed.flags.writeable = False
            session.agent.cache.clear()
        environment.restore(payload['environment'])
        return session
