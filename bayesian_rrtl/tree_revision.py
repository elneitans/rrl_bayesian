"""Experimental local revision of incremental RRTL; the default agent is unchanged.

Loss gates are operational diagnostics under dependent RL data, not guarantees
of policy improvement. Validation transitions must be independent of fitting.
"""
from collections import defaultdict, deque
import copy
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.stats import f

from .baseline import CorrectedNode
from .baseline_adapter import RelationalBaselineAgent, normalize_facts
from .baseline_anova import ANOVANode, anova_from_stats
from .targets import bootstrap_mask, q_learning_update
from relational_regresion_tree import Node

MODES = ('normal', 'frozen', 'prune', 'prune_regrow', 'replace')


@dataclass(frozen=True)
class RevisionConfig:
    mode: str = 'normal'
    buffer_capacity: int = 8192
    min_train: int = 128
    min_validation: int = 64
    min_validation_blocks: int = 4
    min_child: int = 16
    max_candidates: int = 4
    replacement_depth: int = 2
    cooldown: int = 10000
    noninferiority_margin: float = .0025
    replacement_gain: float = .001
    bootstrap_samples: int = 400
    confidence: float = .95
    seed: int = 902711

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError('Unknown revision mode')
        for key in ('buffer_capacity', 'min_train', 'min_validation', 'min_validation_blocks',
                    'min_child', 'max_candidates', 'replacement_depth', 'cooldown', 'bootstrap_samples', 'seed'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError('Positive revision counts required')
        if self.buffer_capacity < self.min_train or not .5 < self.confidence < 1:
            raise ValueError('Invalid revision window/confidence')
        if any(not math.isfinite(x) or x < 0 for x in (self.noninferiority_margin, self.replacement_gain)):
            raise ValueError('Invalid loss margins')


def categories(literal):
    return ('more', 'same', 'less') if literal.type == 'comparative' else ('True', 'False')


def fact_value(state, literal):
    return next((fact.value for fact in state if
                 (fact.name, fact.obj1, fact.obj2) == (literal.name, literal.obj1, literal.obj2)), None)


def route(root, state):
    path, node, visited = 'root', root, []
    while True:
        visited.append((path, node))
        if node.literal is None:
            return visited
        value = fact_value(state, node.literal)
        if value is None:
            return visited  # Preserve internal-node fallback for missing literals.
        path += '/' + value.lower()
        node = getattr(node, value.lower() + '_branch')


def walk(root, path='root'):
    yield path, root
    if root.literal is not None:
        for label in categories(root.literal):
            yield from walk(getattr(root, label.lower() + '_branch'), path + '/' + label.lower())


def leaf_count(node):
    return sum(child.literal is None for _, child in walk(node))


def moments_add(stats, value):
    n = stats['n'] = stats['n'] + 1
    delta = value - stats['mu']
    stats['mu'] += delta / n
    stats['J'] += delta * (value - stats['mu'])


def empty_moments():
    return {'n': 0, 'mu': 0., 'J': 0.}


def score_groups(groups):
    """Both nominal tests and effect size, on the very same observations."""
    total = groups['Total']
    within = sum(g['J'] for k, g in groups.items() if k != 'Total')
    old = float(f.cdf(within / total['J'], total['n']-1, total['n']-1)) if total['J'] > 0 and total['n'] > 1 else None
    result = anova_from_stats(groups)
    return {'n': result.n, 'observed_categories': result.k, 'p_historical': old,
            'p_anova': result.p_value, 'effect_fraction': result.between/(result.within+result.between)
            if result.within+result.between else 0., 'groups': groups}


def paired_loss_gate(differences, blocks, config, rng, *, replacement=False):
    """Episode/block bootstrap, with equal weight for each available block.

    Repeated candidate selection and shared starts prevent interpreting this as
    a calibrated repeated-test guarantee. Return all quantities for auditing.
    """
    values = np.asarray(differences, float)
    ids = np.asarray(blocks)
    if not np.isfinite(values).all() or len(values) != len(ids):
        raise ValueError('Invalid paired losses')
    unique = np.unique(ids)
    means = np.array([values[ids == b].mean() for b in unique])
    result = {'samples': len(values), 'blocks': len(unique), 'accepted': False,
              'mean_difference': float(means.mean()) if len(means) else None}
    if len(values) < config.min_validation or len(unique) < config.min_validation_blocks:
        return {**result, 'reason': 'insufficient_validation'}
    samples = rng.integers(0, len(means), size=(config.bootstrap_samples, len(means)))
    upper = float(np.quantile(means[samples].mean(axis=1), config.confidence))
    threshold = -config.replacement_gain if replacement else config.noninferiority_margin
    return {**result, 'upper_bound': upper, 'threshold': threshold,
            'accepted': upper < threshold, 'reason': 'loss_gate',
            'scope': 'exploratory_block_bootstrap_not_sequential_guarantee'}


class RevisionAgent(RelationalBaselineAgent):
    """A diagnostic fork of a baseline, with its own checkpoint schema/runner."""

    @classmethod
    def fork(cls, agent, revision):
        if not isinstance(agent, RelationalBaselineAgent):
            raise ValueError('Only relational baseline parents are supported')
        result = cls.__new__(cls)
        vars(result).update(copy.deepcopy(vars(agent)))
        result.revision = revision
        result.revision_rng = np.random.default_rng(revision.seed)
        result.buffer = deque(maxlen=revision.buffer_capacity)
        result.events = []
        result.audit = {}
        result.next_uid = 0
        result.structure_version = 0
        result.initial_splits = agent.n_splits
        result.grows_total = result.prunes_total = result.replacements_total = 0
        result.cooldowns = {}
        result.last_detail = None
        # A new fork may itself start from an instrumented model. Its node IDs
        # belong to the old experiment; assign a fresh, internally consistent set.
        for root in result.root_nodes.values():
            for _, node in walk(root):
                if hasattr(node, 'revision_uid'):
                    del node.revision_uid
        result._assign_uids()
        return result

    def _assign_uids(self):
        for root in self.root_nodes.values():
            for _, node in walk(root):
                if not hasattr(node, 'revision_uid'):
                    node.revision_uid = self.next_uid
                    self.next_uid += 1

    def _q(self, state):
        return np.array([root.predict(state) for root in self.root_nodes.values()], dtype=float)

    def _bootstrap_value(self, state):
        """Default online-Q target; experimental subclasses can supply a lagged target."""
        return max(self._q(state))

    def _audit_observation(self, node, state, td):
        # Separate moments for each signal; absent is explicit in diagnostic data.
        for literal in node.refinements:
            key = (node.revision_uid, str(literal))
            entry = self.audit.setdefault(key, {'node': node.revision_uid, 'action': node.root_action,
                'literal': str(literal), 'depth': node.depth, 'signals': {}})
            value = fact_value(state, literal)
            for signal, observation in [('updated_q', td.new_q), ('target', td.target),
                                        ('td_error', td.target-td.old_q)]:
                groups = entry['signals'].setdefault(signal, {'Total': empty_moments(),
                    **{label: empty_moments() for label in categories(literal)}})
                if value is None:
                    # Missing is reported separately, never folded into False.
                    entry['missing'] = entry.get('missing', 0) + int(signal == 'updated_q')
                    continue
                moments_add(groups['Total'], observation)
                moments_add(groups[value], observation)

    def observe(self, state, action, reward, next_state, terminated, truncated):
        normalized = self._state(state)
        if type(action) is not int or not 0 <= action < len(self.actions):
            raise ValueError('Invalid action')
        if not math.isfinite(reward) or type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError('Invalid transition')
        mask = bootstrap_mask(terminated, truncated,
            bootstrap_on_truncation=self.config.bootstrap_on_truncation,
            final_observation_available=next_state is not None)
        following = self._state(next_state) if mask else ()
        root = self.root_nodes[self.actions[action]]
        visited = route(root, normalized)
        path, leaf = visited[-1]
        if type(leaf) is Node:
            leaf.__class__ = CorrectedNode
        self._record_queries(normalized, [root], 'update')
        if mask:
            self._record_queries(following, self.root_nodes.values(), 'bootstrap')
        before_q = self._q(normalized)
        td = q_learning_update(leaf.q_value, reward, lambda: self._bootstrap_value(following),
            eta_q=self.config.eta_q, gamma=self.config.gamma, terminated=terminated,
            truncated=truncated, bootstrap_on_truncation=self.config.bootstrap_on_truncation)
        leaf.q_value = td.new_q
        root.update_statistics(normalized)
        self._audit_observation(leaf, normalized, td)
        allowed = self.revision.mode not in ('frozen', 'prune')
        allowed &= self.interactions >= self.cooldowns.get(leaf.revision_uid, -1)
        was_leaf = leaf.literal is None
        fallback = not was_leaf
        uid = leaf.revision_uid
        if allowed and root.split_iteration(normalized, inherit_q_values=self.config.inherit_q_values):
            self._learner.n_splits += 1
            self.grows_total += 1
            self.structure_version += 1
            self.events.append({'kind': 'grow', 'counter': self.interactions+1,
                'action': self.actions[action], 'path': path, 'node': uid,
                'literal': str(leaf.literal), 'test': self.config.split_test,
                'evidence': score_groups(copy.deepcopy(leaf.refinements_stats[leaf.literal])), 'inherited_q': td.new_q})
            self._assign_uids()
        self.interactions += 1
        encoded = self.encoder.transform([normalized]).values[0].tolist()
        self.last_detail = {'leaf': uid, 'path': path, 'structure_version': self.structure_version,
            'encoded_state': encoded, 'q_before': before_q.tolist(), 'q_after': self._q(normalized).tolist(),
            'greedy_margin': float(np.sort(before_q)[-1]-np.sort(before_q)[-2]) if len(before_q)>1 else 0.,
            'internal_fallback': fallback}
        self.buffer.append({'state': normalized, 'next_state': following, 'action': action,
            'reward': float(reward), 'terminated': terminated, 'truncated': truncated})
        return td

    def _new_node(self, old, *, refinements=None):
        kind = ANOVANode if self.config.split_test == 'anova' else CorrectedNode
        return kind(root_action=old.root_action, refinements=list(old.refinements if refinements is None else refinements),
            max_depth=old.max_depth, min_sample_size=old.min_sample_size, depth=old.depth,
            node_type=old.node_type, parent=old.parent, conjuntion_str=old.conjuntion_str,
            significance_level=old.significance_level, best_literal_criteria=old.best_literal_criteria,
            valid_actions=list(old.valid_actions))

    def _targets(self, records):
        values = []
        for row in records:
            mask = bootstrap_mask(row['terminated'], row['truncated'],
                bootstrap_on_truncation=self.config.bootstrap_on_truncation,
                final_observation_available=row['next_state'] is not None)
            values.append(row['reward'] + self.config.gamma*self._bootstrap_value(row['next_state']) if mask else row['reward'])
        return np.asarray(values, float)

    def _alternative(self, original, records, targets, *, forbidden_root=None):
        """Bounded local SSE reconstruction, followed by original incremental TD.

        Temporary target moments are NOT installed as updated-Q split moments.
        All new nodes start with fresh original-format statistics.
        """
        def build(template, rows, ys, depth_left, forbidden=None):
            node = self._new_node(template)
            node.q_value = float(ys.mean())
            if depth_left == 0 or node.depth >= node.max_depth or len(ys) < 2*self.revision.min_child:
                return node
            best = None
            for literal in node.refinements:
                if literal == forbidden:
                    continue
                vals = [fact_value(r['state'], literal) for r in rows]
                indices = {label: np.array([i for i, v in enumerate(vals) if v == label], int)
                           for label in categories(literal)}
                observed = [ix for ix in indices.values() if len(ix)]
                if len(observed) < 2 or min(map(len, observed)) < self.revision.min_child:
                    continue
                # Missing values use the new internal Q, as in historical routing.
                loss = sum(float(np.sum((ys[ix]-ys[ix].mean())**2)) for ix in observed)
                absent = np.array([i for i, v in enumerate(vals) if v is None], int)
                loss += float(np.sum((ys[absent]-node.q_value)**2))
                if best is None or loss < best[0]:
                    best = loss, literal, indices
            if best is None or best[0] >= float(np.sum((ys-node.q_value)**2))-1e-12:
                return node
            _, literal, indices = best
            node.literal = literal
            for label in categories(literal):
                child = self._new_node(node, refinements=[x for x in node.refinements if x != literal])
                child.parent, child.depth, child.node_type = node, node.depth+1, label
                child.conjuntion_str = node.conjuntion_str+str(literal)+label
                ix = indices[label]
                child = build(child, [rows[i] for i in ix], ys[ix], depth_left-1) if len(ix) else child
                if not len(ix):
                    child.q_value = node.q_value
                setattr(node, label.lower()+'_branch', child)
                node.children.append(child)
            return node
        return build(original, records, targets, self.revision.replacement_depth, forbidden_root)

    def _install(self, action, old, candidate):
        parent = old.parent
        if parent is None:
            self._learner.root_nodes[action] = candidate
        else:
            name = old.node_type.lower()+'_branch'
            if getattr(parent, name) is not old:
                raise ValueError('Revision parent mismatch')
            setattr(parent, name, candidate)
            parent.children = [getattr(parent, k.lower()+'_branch') for k in categories(parent.literal)]
        self._learner.n_splits = sum(n.literal is not None for root in self.root_nodes.values() for _, n in walk(root))
        self._assign_uids()
        for _, node in walk(candidate):
            self.cooldowns[node.revision_uid] = self.interactions+self.revision.cooldown
        self.structure_version += 1

    def maintain(self, validation):
        """One accepted change at most; holdout records must never train the agent."""
        if self.revision.mode in ('normal', 'frozen'):
            return []
        proposals = []
        train_index, valid_index = defaultdict(list), defaultdict(list)
        training = list(self.buffer)
        target_index = {id(r): y for r,y in zip(training, self._targets(training))}
        for records, index in ((training, train_index), (validation, valid_index)):
            for record in records:
                root = self.root_nodes[self.actions[record['action']]]
                for _, node in route(root, record['state']):
                    index[node.revision_uid].append(record)
        for action, root in self.root_nodes.items():
            ai = self.actions.index(action)
            for path, node in walk(root):
                if node.literal is None or self.interactions < self.cooldowns.get(node.revision_uid, -1):
                    continue
                train = train_index[node.revision_uid]
                valid = valid_index[node.revision_uid]
                if len(train) >= self.revision.min_train and len(valid) >= self.revision.min_validation:
                    ys = np.array([target_index[id(r)] for r in train])
                    current_loss = np.mean([(node.predict(r['state'])-y)**2 for r,y in zip(train,ys)])
                    merged_loss = float(np.mean((ys-ys.mean())**2))
                    leaves = leaf_count(node)
                    # Screen using training data only. Avoid always reviewing the
                    # four roots and never considering a small removable branch.
                    priority = -current_loss if self.revision.mode == 'replace' else (merged_loss-current_loss)/max(1,leaves-1)
                    proposals.append((leaves, action, path, node, train, valid, priority))
        proposals.sort(key=lambda p: (p[6], p[1], p[2]))
        results = []
        for leaves, action, path, node, train, valid, _ in proposals[:self.revision.max_candidates]:
            ytrain, yvalid = self._targets(train), self._targets(valid)
            replacement = self.revision.mode == 'replace'
            candidate = self._alternative(node, train, ytrain, forbidden_root=node.literal) if replacement else self._new_node(node)
            if not replacement:
                candidate.q_value = float(ytrain.mean())
            if replacement and candidate.literal is None:
                continue  # Replacement is distinct from pruning.
            current_loss = np.array([(node.predict(r['state'])-y)**2 for r, y in zip(valid, yvalid)])
            new_loss = np.array([(candidate.predict(r['state'])-y)**2 for r, y in zip(valid, yvalid)])
            gate = paired_loss_gate(new_loss-current_loss, [r['block'] for r in valid],
                self.revision, self.revision_rng, replacement=replacement)
            record = {'kind': 'replacement_review' if replacement else 'prune_review',
                'counter': self.interactions, 'action': action, 'path': path, 'node': node.revision_uid,
                'old_literal': str(node.literal), 'new_literal': str(candidate.literal),
                'old_leaves': leaves, 'new_leaves': leaf_count(candidate),
                'train_samples': len(train), 'current_mse': float(current_loss.mean()),
                'candidate_mse': float(new_loss.mean()), 'gate': gate}
            if gate['accepted']:
                self._install(action, node, candidate)
                if replacement:
                    self.replacements_total += 1
                else:
                    self.prunes_total += 1
                record['structure_version'] = self.structure_version
            self.events.append(record)
            results.append(record)
            if gate['accepted']:
                break
        return results

    def audit_report(self, *, reset=False):
        result = []
        for entry in self.audit.values():
            result.append({**{k: v for k, v in entry.items() if k != 'signals'},
                           'signals': {signal: score_groups(groups) for signal, groups in entry['signals'].items()}})
        if reset:
            self.audit.clear()
        return result

    def revision_metrics(self):
        return {'mode': self.revision.mode, 'active_splits': self.n_splits,
            'initial_splits': self.initial_splits, 'grows_total': self.grows_total,
            'prunes_total': self.prunes_total, 'replacements_total': self.replacements_total,
            'structure_version': self.structure_version,
            'leaves': sum(leaf_count(root) for root in self.root_nodes.values())}

    def validate_revision(self):
        from .baseline_checkpoint import _validate_trees
        _validate_trees(self)
        ids = [n.revision_uid for root in self.root_nodes.values() for _, n in walk(root)]
        if len(set(ids)) != len(ids) or self.next_uid <= max(ids):
            raise ValueError('Invalid revision node identities')
        if len(self.buffer) > self.revision.buffer_capacity:
            raise ValueError('Revision buffer overflow')

    def tree_export(self):
        return [{'action': action, 'path': path, 'node': node.revision_uid, 'literal':
                 None if node.literal is None else dict(node.literal._asdict()),
                 'q': node.q_value, 'depth': node.depth, 'leaf': node.literal is None}
                for action, root in self.root_nodes.items() for path, node in walk(root)]
