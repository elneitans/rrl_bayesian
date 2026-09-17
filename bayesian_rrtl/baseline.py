"""Corrected incremental RRTL. The historical module remains byte-for-byte intact."""

import copy
import csv
import json
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np

from relational_regresion_tree import Node, RRLAgent
from .targets import q_learning_update


class FireReset(gym.Wrapper):
    """Historical FIRE/second-action startup with current reset observations."""

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        meanings = self.env.unwrapped.get_action_meanings()
        if len(meanings) < 3 or meanings[1] != "FIRE":
            raise ValueError("FireReset requires FIRE at action 1 and a second startup action")
        for action in (1, 2):
            observation, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                observation, info = self.env.reset()
        return observation, info


class RecordReward(gym.Wrapper):
    def __init__(self, env, game):
        super().__init__(env)
        self.game = game

    def step(self, action):
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info, raw_reward=float(raw_reward))
        reward = float(np.sign(raw_reward)) + (0.1 if self.game == "Pong" else 0.0)
        return observation, reward, terminated, truncated, info


class CorrectedNode(Node):
    """Keep the historical split rule, excluding undefined variance ratios."""

    def find_best_literal(self):
        refinements = self.refinements
        self.refinements = [literal for literal in refinements
                            if self.refinements_stats[literal]['Total']['J'] > 0]
        try:
            return super().find_best_literal()
        finally:
            # Deferred candidates remain available to this leaf and future children.
            self.refinements = refinements


class CorrectedRRLAgent(RRLAgent):
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = Path(output_dir)
        super().__init__(
            env_name=config.resolved_env_id, game_name=config.game, run=config.run,
            save_dir=str(self.output_dir) + "/", initial_seed=config.seed,
            alpha=config.eta_q, gamma=config.gamma, epsilon_init=config.epsilon_init,
            epsilon_min=config.epsilon_min, epsilon_decay_steps=config.epsilon_decay_steps,
            max_depth=config.max_depth, min_sample_size=config.min_sample_size,
            significance_level=config.significance_level,
            action_buffer_capacity=config.action_buffer_capacity,
            best_literal_criteria=config.best_literal_criteria, splits=config.model_version,
            include_incomplete_states=config.include_incomplete_states,
            inherit_q_values=config.inherit_q_values)
        exploration_seed, environment_seed = np.random.SeedSequence(config.seed).spawn(2)
        self.rng = np.random.default_rng(exploration_seed)
        self.environment_rng = np.random.default_rng(environment_seed)
        self.i_step = self.i_episode = self.n_splits = 0
        self.training_seeds = []

    def _environment(self, training):
        env = gym.make(self.env_name, obs_type="rgb", frameskip=self.config.frameskip,
                       repeat_action_probability=self.config.repeat_action_probability,
                       full_action_space=False, max_episode_steps=self.config.max_episode_steps)
        if env.unwrapped.get_action_meanings() != self.valid_actions:
            env.close()
            raise ValueError("ALE actions differ from the historical action catalog")
        return FireReset(RecordReward(env, self.game_name) if training else env)

    def make_train_environment(self):
        return self._environment(True)

    def make_test_environment(self):
        return self._environment(False)

    def _episode_seed(self):
        reserved = set(self.config.validation_seeds + self.config.test_seeds)
        while True:
            seed = int(self.environment_rng.integers(0, 2**32))
            if seed not in reserved:
                self.training_seeds.append(seed)
                return seed

    def update_transition(self, state, action_index, reward, next_state, terminated, truncated):
        root = self.root_nodes[self.index_to_action[action_index]]
        leaf = root.get_state_leaf(state)
        # Legacy factories and older checkpoints contain Node instances. Specialize
        # only the active leaf, preserving its identity, parent links and statistics.
        if type(leaf) is Node:
            leaf.__class__ = CorrectedNode
        update = q_learning_update(
            leaf.q_value, reward,
            lambda: max(node.predict(next_state) for node in self.root_nodes.values()),
            eta_q=self.config.eta_q, gamma=self.gamma, terminated=terminated, truncated=truncated,
            bootstrap_on_truncation=self.config.bootstrap_on_truncation)
        leaf.q_value = update.new_q
        root.update_statistics(state)
        if root.split_iteration(state, inherit_q_values=self.inherit_q_values):
            self.n_splits += 1
        return update

    def train(self):
        if self.i_step:
            raise ValueError("H0 supports fresh training and checkpoint evaluation, not mid-episode resume")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "train.csv"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        env = self.make_train_environment()
        last_episode = {}
        try:
            with path.open("x", newline="") as handle:
                fields = ["Game", "Run", "Episode", "Iteration", "Reward", "Raw reward",
                          "Episode return", "Raw episode return", "Number of splits", "Action",
                          "Terminated", "Truncated", "Bootstrap", "TD target", "Old Q", "New Q",
                          "Epsilon", "Episode seed", "Budget exhausted", "Empty state"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                while self.i_step < self.config.n_iterations:
                    seed = self._episode_seed()
                    observation, _ = env.reset(seed=seed)
                    self.action_buffer.clear()
                    previous_info = self.preprocessor.get_info(observation)
                    state = self.get_state(previous_info, None, self.include_incomplete_states)
                    episode_return = raw_return = 0.0
                    while self.i_step < self.config.n_iterations:
                        epsilon = self.epsilon
                        action = self.epsilon_greedy_policy(state)
                        if self.game_name == "Breakout" and self.config.resample_repeated_actions:
                            action = self.resample_repeated_action(action)
                        observation, reward, terminated, truncated, env_info = env.step(action)
                        # This runner uses non-autoresetting Gymnasium environments: step()
                        # returns the actual final observation for truncations.
                        info = self.preprocessor.get_info(observation)
                        next_state = self.get_state(info, previous_info, self.include_incomplete_states)
                        update = self.update_transition(state, action, reward, next_state, terminated, truncated)
                        raw_reward = float(env_info["raw_reward"])
                        episode_return += reward
                        raw_return += raw_reward
                        self.i_step += 1
                        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)
                        done = bool(terminated or truncated)
                        exhausted = self.i_step == self.config.n_iterations
                        writer.writerow(dict(zip(fields, [
                            self.game_name, self.run, self.i_episode, self.i_step, reward, raw_reward,
                            episode_return, raw_return, self.n_splits, action, bool(terminated),
                            bool(truncated), update.bootstrap, update.target, update.old_q, update.new_q,
                            epsilon, seed, exhausted, not bool(state)])))
                        last_episode = {"return": episode_return, "raw_return": raw_return,
                                        "complete": done, "seed": seed}
                        state, previous_info = next_state, info
                        if done:
                            self.i_episode += 1
                            break
            checkpoint = self.save_final_checkpoint(last_episode)
            summary = {"variant": self.config.variant, "iterations": self.i_step,
                       "completed_episodes": self.i_episode, "splits": self.n_splits,
                       "last_episode": last_episode, "training_seeds": self.training_seeds,
                       "checkpoint": checkpoint.name, "config_hash": self.config.config_hash}
            (self.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
            if self.config.render_graph:
                self.get_dot_representation().render(str(self.output_dir / "final_tree"), view=False)
            return summary
        finally:
            env.close()

    def save_final_checkpoint(self, last_episode):
        payload = {"schema_version": 1, "variant": self.config.variant,
                   "config": self.config.to_dict(), "config_hash": self.config.config_hash,
                   "root_nodes": self.root_nodes, "epsilon": self.epsilon,
                   "i_step": self.i_step, "i_episode": self.i_episode, "n_splits": self.n_splits,
                   "rng": self.rng.bit_generator.state,
                   "environment_rng": self.environment_rng.bit_generator.state,
                   "training_seeds": self.training_seeds, "last_episode": last_episode,
                   "action_buffer": self.action_buffer,
                   "purpose": "final model evaluation; emulator state not included"}
        path = self.output_dir / "final_checkpoint.pkl"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        return path

    def load_final_checkpoint(self, path):
        """Load trusted local checkpoints for evaluation, not arbitrary pickle files."""
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if payload["schema_version"] != 1 or payload["config_hash"] != self.config.config_hash:
            raise ValueError("Checkpoint schema/config mismatch")
        self.root_nodes = payload["root_nodes"]
        self.epsilon = payload["epsilon"]
        for name in ("i_step", "i_episode", "n_splits", "training_seeds", "action_buffer"):
            setattr(self, name, payload[name])
        self.rng.bit_generator.state = payload["rng"]
        self.environment_rng.bit_generator.state = payload["environment_rng"]

    def evaluate(self, seeds):
        """Evaluate a private copy, preserving training RNG, buffers and tree statistics."""
        if set(seeds) & set(self.training_seeds):
            raise ValueError("Evaluation seeds overlap training episodes")
        agent = copy.deepcopy(self)
        env = agent.make_test_environment()
        rows = []
        try:
            for seed in seeds:
                agent.rng = np.random.default_rng(seed)
                agent.action_buffer.clear()
                observation, _ = env.reset(seed=seed)
                previous_info = agent.preprocessor.get_info(observation)
                state = agent.get_state(previous_info, None, agent.include_incomplete_states)
                total, steps = 0.0, 0
                while True:
                    action = agent.greedy_policy(state)
                    if agent.game_name == "Breakout" and agent.config.resample_repeated_actions:
                        action = agent.resample_repeated_action(action)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    total += reward
                    steps += 1
                    if terminated or truncated:
                        rows.append({"seed": seed, "return": total, "steps": steps,
                                     "terminated": bool(terminated), "truncated": bool(truncated)})
                        break
                    info = agent.preprocessor.get_info(observation)
                    state = agent.get_state(info, previous_info, agent.include_incomplete_states)
                    previous_info = info
            return rows
        finally:
            env.close()
