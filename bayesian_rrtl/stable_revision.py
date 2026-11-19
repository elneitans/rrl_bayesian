"""Prospective RRTL improvements: lagged Double targets and guarded repair.

Operational guards limit observed changes; they are not policy-improvement
guarantees. Neither validation probes nor rollout returns are training data.
"""
from collections import defaultdict
import copy
from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from .tree_revision import RevisionAgent, RevisionConfig, leaf_count, paired_loss_gate, route, walk

VARIANTS = ('baseline', 'slow_td', 'double_td', 'stable_td', 'guarded', 'stable_guarded', 'refit_only')


@dataclass(frozen=True)
class ImprovementConfig:
    variant: str = 'baseline'
    eta_q: float = .025
    bootstrap: str = 'online'
    target_interval: int = 5000
    repair: str = 'none'
    max_policy_tv: float = .05
    complexity_penalty: float = .0001
    return_confidence: float = .9
    return_bootstraps: int = 400
    return_margin: float = 1.
    max_start_drop: float = 5.
    relative_start_drop: float = .2

    def __post_init__(self):
        if self.variant not in VARIANTS or self.bootstrap not in ('online', 'double') or self.repair not in ('none', 'guarded', 'refit'):
            raise ValueError('Invalid improvement variant')
        if not 0 < self.eta_q <= 1 or type(self.target_interval) is not int or self.target_interval < 1:
            raise ValueError('Invalid TD settings')
        if not 0 <= self.max_policy_tv <= 1 or not .5 < self.return_confidence < 1:
            raise ValueError('Invalid policy/return guard')
        if type(self.return_bootstraps) is not int or self.return_bootstraps < 1:
            raise ValueError('Invalid return bootstrap count')
        for name in ('complexity_penalty', 'return_margin', 'max_start_drop', 'relative_start_drop'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError('Invalid nonnegative guard setting')


def variant_config(name, **overrides):
    if name not in VARIANTS:
        raise ValueError('Unknown variant')
    return ImprovementConfig(**{'variant': name,
        'eta_q': .005 if name in ('slow_td', 'stable_td', 'stable_guarded') else .025,
        'bootstrap': 'double' if name in ('double_td', 'stable_td', 'stable_guarded') else 'online',
        'repair': 'guarded' if name in ('guarded', 'stable_guarded') else 'refit' if name == 'refit_only' else 'none',
        **overrides})


def return_gate(before, after, config, rng):
    """Paired start-level operational gate, with explicit catastrophe guard."""
    old, new = np.asarray(before, float), np.asarray(after, float)
    if old.ndim != 1 or old.shape != new.shape or len(old) < 2 or not np.isfinite([old, new]).all():
        raise ValueError('Invalid paired return data')
    differences = new-old
    indices = rng.integers(len(old), size=(config.return_bootstraps, len(old)))
    lower = float(np.quantile(differences[indices].mean(axis=1), 1-config.return_confidence))
    allowed_drop = np.maximum(config.max_start_drop, config.relative_start_drop*np.abs(old))
    accepted = differences.mean() >= 0 and lower >= -config.return_margin and np.all(differences >= -allowed_drop)
    return {'accepted': bool(accepted), 'mean_difference': float(differences.mean()),
        'lower_bound': lower, 'margin': config.return_margin, 'differences': differences.tolist(),
        'allowed_start_drops': allowed_drop.tolist(), 'starts': len(old),
        'scope': 'exploratory_paired_start_gate_not_a_return_guarantee'}


class StableRevisionAgent(RevisionAgent):
    @classmethod
    def fork(cls, agent, revision, improvement):
        if revision.mode != 'normal' and (revision.mode != 'frozen' or improvement.repair != 'none'):
            raise ValueError('Only revision-free TD controls may freeze growth')
        result = super().fork(agent, revision)
        result.improvement = improvement
        result.config = replace(result.config, eta_q=improvement.eta_q)
        result._learner.config = result.config
        result._learner.gamma = result.config.gamma
        result.target_roots = copy.deepcopy(result.root_nodes) if improvement.bootstrap == 'double' else {}
        result.target_counter = result.interactions
        result.target_syncs = 0
        result.refits_total = result.parameter_version = 0
        return result

    def _bootstrap_value(self, state):
        if self.improvement.bootstrap == 'online':
            return super()._bootstrap_value(state)
        online = self._q(state)
        selected = np.flatnonzero(online == online.max())
        return float(np.mean([self.target_roots[self.actions[i]].predict(state) for i in selected]))

    def observe(self, *args, **kwargs):
        used = self.target_counter
        result = super().observe(*args, **kwargs)
        self.last_detail.update(bootstrap_method=self.improvement.bootstrap, target_counter_used=used,
                                parameter_version=self.parameter_version)
        if self.improvement.bootstrap == 'double' and self.interactions-self.target_counter >= self.improvement.target_interval:
            self.target_roots = copy.deepcopy(self.root_nodes)
            self.target_counter = self.interactions
            self.target_syncs += 1
            self.events.append({'kind': 'target_sync', 'counter': self.interactions,
                                'target_counter': self.target_counter, 'syncs': self.target_syncs})
        self.last_detail['target_counter_after'] = self.target_counter
        return result

    def _prediction_agent(self, action, path, candidate):
        """Independent forest for prediction only; no counters/RNG/TD mutations."""
        proxy = copy.copy(self)
        proxy._learner = copy.copy(self._learner)
        proxy._learner.root_nodes = copy.deepcopy(self.root_nodes)
        old = next(node for p, node in walk(proxy.root_nodes[action]) if p == path)
        fresh = copy.deepcopy(candidate)
        fresh.parent = old.parent
        if old.parent is None:
            proxy._learner.root_nodes[action] = fresh
        else:
            setattr(old.parent, old.node_type.lower()+'_branch', fresh)
            old.parent.children = [getattr(old.parent, name.lower()+'_branch') for name in
                (('more', 'same', 'less') if old.parent.literal.type == 'comparative' else ('True', 'False'))]
        return proxy

    def _bounded(self, old, candidate, training, targets):
        reached, endings = defaultdict(list), defaultdict(list)
        for i, row in enumerate(training):
            visited = route(candidate, row['state'])
            for _, node in visited:
                reached[id(node)].append(i)
            endings[id(visited[-1][1])].append(i)
        for _, node in walk(candidate):
            ix = reached[id(node)]
            if not ix:
                node.q_value = node.parent.q_value if node.parent is not None else old.q_value
                continue
            previous = np.array([old.predict(training[i]['state']) for i in ix])
            fit = endings[id(node)]
            desired = float(np.mean(targets[fit])) if len(fit) >= self.revision.min_child else float(previous.mean())
            node.q_value = float(np.clip(desired, previous.min(), previous.max()))
        return candidate

    def _refit(self, old, training, targets):
        candidate = copy.deepcopy(old)
        endings = defaultdict(list)
        for i, row in enumerate(training):
            endings[id(route(candidate, row['state'])[-1][1])].append(i)
        for _, node in walk(candidate):
            ix = endings[id(node)]
            if len(ix) >= self.revision.min_child:
                node.q_value = float(np.mean(targets[ix]))
        return candidate

    def _policy_guard(self, candidate, validation):
        by_block = defaultdict(list)
        changes = []
        for row in validation:
            before, after = self._q(row['state']), candidate._q(row['state'])
            p, q = before == before.max(), after == after.max()
            tv = float(np.abs(p/p.sum()-q/q.sum()).sum()/2)
            by_block[row['block']].append(tv)
            changes.append(tv > 0)
        value = float(np.mean([np.mean(v) for v in by_block.values()]))
        return {'accepted': value <= self.improvement.max_policy_tv, 'mean_tv': value,
                'limit': self.improvement.max_policy_tv, 'samples': len(validation),
                'changed_state_fraction': float(np.mean(changes)), 'blocks': len(by_block)}

    def maintain(self, validation, policy_gate=None):
        if self.improvement.repair == 'none':
            return []
        if not validation:
            raise ValueError('Independent validation required')
        if self.improvement.repair == 'guarded' and policy_gate is None:
            raise ValueError('Guarded repair requires a rollout-return validator')
        training = list(self.buffer)
        targets = self._targets(training)
        ti, vi = defaultdict(list), defaultdict(list)
        for rows, index in ((training, ti), (validation, vi)):
            for i, row in enumerate(rows):
                for _, node in route(self.root_nodes[self.actions[row['action']]], row['state']):
                    index[node.revision_uid].append(i)
        screen = []
        for action, root in self.root_nodes.items():
            for path, node in walk(root):
                ix = ti[node.revision_uid]
                if (node.literal is None or self.interactions < self.cooldowns.get(node.revision_uid, -1)
                    or len(ix) < self.revision.min_train or len(vi[node.revision_uid]) < self.revision.min_validation):
                    continue
                ys = targets[ix]
                old_loss = np.mean([(node.predict(training[i]['state'])-y)**2 for i,y in zip(ix,ys)])
                score = (np.mean((ys-ys.mean())**2)-old_loss)/max(1,leaf_count(node)-1)
                screen.append((float(score), action, path, node, ix))
        proposals = []
        for _, action, path, node, ix in sorted(screen, key=lambda v:v[:3])[:self.revision.max_candidates]:
            rows, ys = [training[i] for i in ix], targets[ix]
            kinds = ('refit',) if self.improvement.repair == 'refit' else ('prune','replacement')
            for kind in kinds:
                candidate = self._refit(node, rows, ys) if kind == 'refit' else self._new_node(node) if kind == 'prune' else self._alternative(node, rows, ys, forbidden_root=node.literal)
                if kind == 'replacement' and candidate.literal is None:
                    continue
                if kind != 'refit':
                    candidate = self._bounded(node, candidate, rows, ys)
                old_loss = np.mean([(node.predict(row['state'])-y)**2 for row,y in zip(rows,ys)])
                new_loss = np.mean([(candidate.predict(row['state'])-y)**2 for row,y in zip(rows,ys)])
                score = float(new_loss-old_loss+self.improvement.complexity_penalty*(leaf_count(candidate)-leaf_count(node)))
                proposals.append((score, action, path, kind, node, candidate, rows))
        result = []
        for _, action, path, kind, node, candidate, train in sorted(proposals, key=lambda v:v[:4])[:self.revision.max_candidates]:
            valid = [validation[i] for i in vi[node.revision_uid]]
            ys = self._targets(valid)  # Same online selector and lagged evaluator until commit.
            old_loss = np.array([(node.predict(row['state'])-y)**2 for row,y in zip(valid,ys)])
            new_loss = np.array([(candidate.predict(row['state'])-y)**2 for row,y in zip(valid,ys)])
            loss = paired_loss_gate(new_loss-old_loss, [row['block'] for row in valid],
                                    self.revision, self.revision_rng, replacement=kind == 'replacement')
            record = {'kind': kind+'_review', 'counter': self.interactions, 'action': action,
                'path': path, 'node': node.revision_uid, 'old_literal': str(node.literal),
                'new_literal': str(candidate.literal), 'old_leaves': leaf_count(node),
                'new_leaves': leaf_count(candidate), 'train_samples': len(train),
                'current_mse': float(old_loss.mean()), 'candidate_mse': float(new_loss.mean()),
                'loss_gate': loss, 'gate': {**loss, 'accepted': False}, 'bounded_q': kind != 'refit'}
            proxy = self._prediction_agent(action, path, candidate)
            changed = any(a.q_value != b.q_value for (_,a),(_,b) in zip(walk(node),walk(candidate))) if kind == 'refit' else True
            if loss['accepted'] and changed:
                trust = self._policy_guard(proxy, validation)
                record['policy_guard'] = trust
                if kind == 'refit':
                    record['gate'] = {**loss, 'accepted': True}
                elif not trust['accepted']:
                    record['gate']['reason'] = 'policy_trust'
                else:
                    rollout_gate = policy_gate(self, proxy, {'action': action, 'path': path, 'kind': kind})
                    record['return_guard'] = rollout_gate
                    record['gate'] = {**loss, 'accepted': bool(rollout_gate['accepted']), 'reason': 'return_guard'}
            elif loss['accepted']:
                record['gate']['reason'] = 'no_parameter_change'
            if record['gate']['accepted']:
                if kind == 'refit':
                    for (_,original),(_,fresh) in zip(walk(node),walk(candidate)):
                        original.q_value = fresh.q_value
                    self.refits_total += 1
                else:
                    self._install(action, node, candidate)
                    if kind == 'prune': self.prunes_total += 1
                    else: self.replacements_total += 1
                self.parameter_version += 1
                record.update(parameter_version=self.parameter_version, structure_version=self.structure_version)
            self.events.append(record);result.append(record)
            if record['gate']['accepted']:
                break
        return result

    def revision_metrics(self):
        return {**super().revision_metrics(), 'mode': self.improvement.variant,
                'target_syncs': self.target_syncs, 'target_counter': self.target_counter,
                'refits_total': self.refits_total, 'parameter_version': self.parameter_version}

    def validate_revision(self):
        super().validate_revision()
        ImprovementConfig(**asdict(self.improvement))
        if self.config.eta_q != self.improvement.eta_q or self.refits_total < 0 or self.parameter_version < 0:
            raise ValueError('Improvement model/config mismatch')
        if self.improvement.bootstrap == 'double':
            if not 0 <= self.interactions-self.target_counter < self.improvement.target_interval:
                raise ValueError('Invalid target snapshot frontier')
            online_ids = {id(n) for root in self.root_nodes.values() for _,n in walk(root)}
            if online_ids & {id(n) for root in self.target_roots.values() for _,n in walk(root)}:
                raise ValueError('Target forest aliases the online model')
            from .baseline_checkpoint import _validate_trees
            target = copy.copy(self);target._learner = copy.copy(self._learner)
            target._learner.root_nodes = self.target_roots
            target._learner.n_splits = sum(n.literal is not None for root in self.target_roots.values() for _,n in walk(root))
            _validate_trees(target)
        elif self.target_roots:
            raise ValueError('Online target variant has a target forest')
