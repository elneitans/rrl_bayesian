"""Sequential, resumable four-cell Pong comparison; test is a separate explicit stage."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

from bayesian_rrtl.atari import AtariConfig
from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig
from bayesian_rrtl.baseline_training import BaselineTrainingSession
from bayesian_rrtl.comparison import BayesianComparisonSession, ComparisonPolicyConfig, evaluation_summary, make_comparison_session
from bayesian_rrtl.comparison_protocol import ROOT, digest, file_hash, resolve_protocol, sources
from bayesian_rrtl.evaluation import learning_curve_area, paired_bootstrap
from bayesian_rrtl.q_config import BayesianQConfig
from experiments.campaign_h7 import write_json_atomic


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, data)


def freeze(raw, output):
    resolved = resolve_protocol(raw)  # all cells validated before any directory/run
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**resolved, 'source_hashes': sources(), 'evaluation_schema': 'pong_evaluation_v2'}
    write(output / 'protocol.json', manifest)
    (output / 'protocol.sha256').write_text(file_hash(output / 'protocol.json'))
    return manifest


def load_frozen(output, *, check_sources=True):
    output = Path(output)
    if file_hash(output / 'protocol.json') != (output / 'protocol.sha256').read_text():
        raise ValueError('Frozen protocol changed')
    manifest = read(output / 'protocol.json')
    if digest(resolve_protocol(manifest['protocol'])) != digest({k: manifest[k] for k in
                                                               ('protocol', 'cells', 'execution_order')}):
        raise ValueError('Resolved protocol mismatch')
    if check_sources and manifest['source_hashes'] != sources():
        raise ValueError('Training source changed; incompatible resume/evaluation')
    return manifest


def operational_budgets(output, manifest):
    """Validate an append-only revision history; experimental identity stays frozen."""
    limits = dict(manifest['protocol']['budgets'])
    path = Path(output) / 'operational_budgets.json'
    if not path.exists():
        return limits, []
    record = read(path)
    if record.get('protocol_hash') != digest(manifest):
        raise ValueError('Operational budget protocol mismatch')
    previous = digest(manifest)
    revisions = record['revisions']
    for index, revision in enumerate(revisions, 1):
        content = {k: v for k, v in revision.items() if k != 'hash'}
        if (revision['hash'] != digest(content) or revision['previous_hash'] != previous
                or revision['revision'] != index or revision['before'] != limits):
            raise ValueError('Invalid operational budget history')
        after = revision['after']
        if (set(after) != set(limits) or any(type(after[k]) not in (int, float)
                or not math.isfinite(after[k]) or after[k] < limits[k] for k in limits)
                or after == limits or not revision['reason'].strip()):
            raise ValueError('Operational budgets must increase explicitly')
        limits, previous = after, revision['hash']
    return limits, revisions


def extend_budgets(output, manifest, *, training_seconds=None, evaluation_seconds=None, reason=None):
    limits, revisions = operational_budgets(output, manifest)
    after = dict(limits)
    for key, value in (('training_seconds_per_run', training_seconds),
                       ('evaluation_seconds_total', evaluation_seconds)):
        if value is not None:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= limits[key]:
                raise ValueError('New operational budget must be finite and greater than its current limit')
            after[key] = value
    if after == limits or not isinstance(reason, str) or not reason.strip():
        raise ValueError('Budget extension requires increased limits and an explicit reason')
    consumed = {}
    for cell in manifest['cells']:
        folder = Path(output) / cell['id']
        if (folder / 'status.json').exists():
            status = _validate_run(manifest, cell, folder)
            consumed[cell['id']] = {k: status.get(k, 0.) for k in
                                    ('interactions', 'training_seconds', 'evaluation_seconds')}
    revision = {'revision': len(revisions) + 1, 'previous_hash': revisions[-1]['hash'] if revisions else digest(manifest),
                'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'reason': reason.strip(),
                'before': limits, 'after': after, 'consumed': consumed}
    revision['hash'] = digest(revision)
    write(Path(output) / 'operational_budgets.json',
          {'protocol_hash': digest(manifest), 'revisions': revisions + [revision]})
    return after


def config_for(cell):
    return (BaselineLearnerConfig if cell['learner'] == 'corrected_common' else BayesianQConfig)(**cell['config'])


def load_session(cell, path):
    if cell['learner'] == 'corrected_common':
        return BaselineTrainingSession.load(path, expected_config=config_for(cell),
            expected_policy=ComparisonPolicyConfig(**cell['policy']),
            expected_environment=AtariConfig(**cell['environment']))
    return BayesianComparisonSession.load(path, expected_config=config_for(cell))


def binding(manifest, cell, path, split, seeds, horizon):
    return {**({'evaluation_schema': manifest['evaluation_schema']} if 'evaluation_schema' in manifest else {}),
            'protocol_hash': digest(manifest), 'cell_hash': cell['hash'], 'checkpoint_sha256': file_hash(path),
            'evaluation_sources_hash': digest(manifest['source_hashes']), 'split': split,
            'seeds': list(seeds), 'horizon': horizon}


def validate_evaluation(saved, expected):
    if saved.get('binding') != expected:
        raise ValueError('Evaluation checkpoint/protocol/source binding mismatch')
    report = saved.get('report', {})
    if 'evaluation_schema' in expected and report.get('schema') != expected['evaluation_schema']:
        raise ValueError('Evaluation schema mismatch')
    rows = report.get('episodes', [])
    if len(rows) != len(expected['seeds']) or report.get('horizon') != expected['horizon']:
        raise ValueError('Incomplete evaluation results')
    for row, seed in zip(rows, expected['seeds']):
        if (row.get('seed') != seed or not isinstance(row.get('return'), (int, float))
                or not math.isfinite(row['return']) or type(row.get('steps')) is not int
                or not 0 < row['steps'] <= expected['horizon']
                or any(type(row.get(flag)) is not bool for flag in ('terminated', 'truncated'))
                or not (row['terminated'] or row['truncated']) or row.get('epsilon') != 0.
                or row.get('return_scope') != ('complete_game' if row['terminated'] else 'horizon_limited')):
            raise ValueError('Invalid evaluation episode')
    if (report.get('mean_raw_return') != float(np.mean([r['return'] for r in rows]))
            or report.get('truncation_fraction') != sum(r['truncated'] for r in rows) / len(rows)):
        raise ValueError('Invalid evaluation aggregate')
    if report.get('schema') is not None:
        if report['schema'] != 'pong_evaluation_v2':
            raise ValueError('Unknown evaluation schema')
        for row in rows:
            if (any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
                    or row[k] < 0 for k in ('points_won', 'points_lost'))
                    or not math.isclose(row['points_won'] - row['points_lost'], row['return'], abs_tol=1e-9)):
                raise ValueError('Invalid evaluation point counts')
        expected_report = evaluation_summary(rows, expected['horizon'])
        if any(report.get(key) != value for key, value in expected_report.items()):
            raise ValueError('Invalid evaluation statistics')


def evaluate_saved(manifest, cell, checkpoint, split, *, evaluation_budget=None):
    p = manifest['protocol']
    seeds = p[f'{split}_seeds']
    expected = binding(manifest, cell, checkpoint, split, seeds, p['evaluation_horizon'])
    session = load_session(cell, checkpoint)
    start = time.perf_counter()
    rows = []
    try:
        for index, seed in enumerate(seeds):
            if evaluation_budget is not None and time.perf_counter() - start >= evaluation_budget:
                raise TimeoutError('Evaluation budget exhausted; completed checkpoints preserved')
            report = session.evaluate((seed,), horizon=p['evaluation_horizon'])
            row = report['episodes'][0]
            row['episode_id'] = index
            rows.append(row)
    finally:
        session.environment.close()
    report = evaluation_summary(rows, p['evaluation_horizon'])
    if file_hash(checkpoint) != expected['checkpoint_sha256']:
        raise ValueError('Evaluation mutated checkpoint')
    saved = {'binding': expected, 'report': report, 'seconds': time.perf_counter() - start}
    validate_evaluation(saved, expected)
    return saved


def run_metrics(session):
    agent = session.agent
    common = session.metrics()
    actions = agent.actions if hasattr(agent, 'actions') else agent.config.actions
    common['action_coverage'] = {name: sum(row['action'] == i for row in session.history)
                                 for i, name in enumerate(actions)}
    common['points_won'] = sum(row['raw_reward'] > 0 for row in session.history)
    common['points_lost'] = sum(row['raw_reward'] < 0 for row in session.history)
    common['environment_profile'] = session.environment.profile
    common['complexity'] = {}
    if hasattr(agent, 'root_nodes'):
        for name, root in agent.root_nodes.items():
            stack, nodes = [root], []
            while stack:
                node = stack.pop()
                nodes.append(node)
                stack.extend(node.children)
            leaves = [n for n in nodes if n.literal is None]
            common['complexity'][name] = {'nodes': len(nodes), 'leaves': len(leaves),
                'max_depth': max(n.depth for n in nodes), 'splits': len(nodes) - len(leaves),
                'eligible_leaf_candidates': sum(stats['Total']['n'] > n.min_sample_size
                    and stats['Total']['J'] > 0 for n in leaves for stats in n.refinements_stats.values()),
                'max_candidate_samples': max((s['Total']['n'] for n in leaves
                                              for s in n.refinements_stats.values()), default=0),
                'max_abs_q': max(abs(n.q_value) for n in nodes)}
        common['targets_outside_reward_scale'] = sum(
            not session.environment.config.reward_bounds[0] / (1 - agent.config.gamma)
            <= r['learner_metrics']['td']['target'] <= session.environment.config.reward_bounds[1] / (1 - agent.config.gamma)
            for r in session.history)
    else:
        reference = agent.encoder.transform([session.state])
        common['max_abs_q'] = float(np.abs(agent.model.predict(reference)).max())
        common['incomplete_replay_states'] = sum(not t.state.observed.all() for t in agent.replay.transitions)
        common['empty_replay_states'] = sum(not t.state.observed.any() for t in agent.replay.transitions)
        common['targets_outside_reward_scale'] = sum(
            d.get('targets_outside_scale', 0) for f in agent.fit_history for d in f['actions'].values())
        for name, posterior in zip(actions, agent.model.posteriors):
            if posterior is not None:
                common['complexity'][name] = {'trees': posterior.m,
                    'leaves_last_draw': posterior.trace[-1].total_leaves,
                    'max_depth_last_draw': posterior.trace[-1].max_depth,
                    'leaves_draw_range': [min(t.total_leaves for t in posterior.trace[posterior.burn_in:]),
                                          max(t.total_leaves for t in posterior.trace[posterior.burn_in:])]}
    common['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    return common


def _validate_run(manifest, cell, folder, *, complete=False):
    status = read(folder / 'status.json')
    if status.get('cell_hash') != cell['hash'] or status.get('protocol_hash') != digest(manifest):
        raise ValueError('Run identity mismatch')
    if (folder / 'manifest.json').exists():
        if read(folder / 'manifest.json')['cell'] != json.loads(json.dumps(cell)):
            raise ValueError('Run manifest mismatch')
    if status.get('checkpoint'):
        if file_hash(folder / status['checkpoint']) != status['checkpoint_sha256']:
            raise ValueError('Progress checkpoint changed')
    if complete and status['state'] != 'completed':
        raise ValueError('All cells must be completed before reporting/test')
    return status


def run_one(output, cell_id, *, evaluation_budget):
    output = Path(output)
    manifest = load_frozen(output)
    cell = next(c for c in manifest['cells'] if c['id'] == cell_id)
    p = manifest['protocol']
    limits, revisions = operational_budgets(output, manifest)
    folder = output / cell_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'status.json').exists():
        status = _validate_run(manifest, cell, folder)
        if status['state'] == 'completed':
            return status
    else:
        status = {'cell_hash': cell['hash'], 'protocol_hash': digest(manifest), 'state': 'partial',
                  'checkpoint': None, 'training_seconds': 0., 'evaluation_seconds': 0., 'save_seconds': 0.}
    status['operational_budget_revision'] = len(revisions)
    status['operational_budgets'] = limits
    session = None
    evaluation_start = status['evaluation_seconds']
    def publish():
        write(folder / 'status.json', status)
    def save():
        start = time.perf_counter()
        relative = f'checkpoint_{session.interactions}.rrtl'
        path = folder / relative
        # A published checkpoint is immutable; it may already be the current point.
        if status.get('checkpoint') != relative:
            session.save(path)
        status.update(checkpoint=relative, checkpoint_sha256=file_hash(path), interactions=session.interactions)
        status['save_seconds'] += time.perf_counter() - start
        write(folder / 'train.json', session.history)
        write(folder / 'episodes.json', session.episode_rows())
        publish()
        return path
    try:
        if status.get('checkpoint'):
            session = load_session(cell, folder / status['checkpoint'])
        else:
            session = make_comparison_session(cell['learner'], AtariConfig(**cell['environment']),
                config_for(cell), policy_config=ComparisonPolicyConfig(**cell['policy']))
            write(folder / 'manifest.json', {'cell': cell, 'learner_id': getattr(session.agent, 'learner_id', 'bart'),
                    'runtime_environment': session.environment.runtime_manifest(), 'source_hashes': manifest['source_hashes']})
            write(folder / 'config.json', cell)
            write(folder / 'encoder.json', session.agent.encoder.to_dict())
            save()
        for point in p['curve_checkpoints']:
            if point < session.interactions:
                if not (folder / f'validation_{point}.json').exists():
                    raise ValueError('Missing evaluation before saved training position')
                continue
            while session.interactions < point:
                if status['training_seconds'] >= limits['training_seconds_per_run']:
                    save()
                    status.update(state='partial', reason='training_budget_exhausted')
                    publish()
                    return status
                start = time.perf_counter()
                session.step()
                status['training_seconds'] += time.perf_counter() - start
                if session.interactions % p['checkpoint_interval'] == 0:
                    save()
            if status['training_seconds'] > limits['training_seconds_per_run']:
                save()
                status.update(state='partial', reason='training_budget_exhausted')
                publish()
                return status
            path = save()
            evaluation_path = folder / f'validation_{point}.json'
            expected = binding(manifest, cell, path, 'validation', p['validation_seeds'], p['evaluation_horizon'])
            if evaluation_path.exists():
                validate_evaluation(read(evaluation_path), expected)
            else:
                remaining = evaluation_budget - (status['evaluation_seconds'] - evaluation_start)
                start = time.perf_counter()
                try:
                    result = evaluate_saved(manifest, cell, path, 'validation', evaluation_budget=remaining)
                    write(evaluation_path, result)
                finally:
                    status['evaluation_seconds'] += time.perf_counter() - start
                    publish()
            print(json.dumps({'cell': cell_id, 'interaction': point, 'validated': True}), flush=True)
        final = folder / 'final_checkpoint.rrtl'
        if not final.exists():
            # Publish exactly the checkpoint bytes already evaluated at the final point.
            temporary = final.with_suffix('.tmp')
            shutil.copyfile(folder / status['checkpoint'], temporary)
            temporary.replace(final)
        if file_hash(final) != status['checkpoint_sha256']:
            raise ValueError('Final checkpoint mismatch')
        curve = [{'interactions': n, 'mean_return': read(folder / f'validation_{n}.json')['report']['mean_raw_return']}
                 for n in p['curve_checkpoints']]
        summary = {'cell': cell, 'curve': curve, 'validation_auc': learning_curve_area(curve),
                   'timing': {key: status[key] for key in ('training_seconds', 'evaluation_seconds', 'save_seconds')},
                   'checkpoint_bytes': final.stat().st_size, 'final_checkpoint_sha256': file_hash(final),
                   'metrics': run_metrics(session), 'state': 'completed'}
        write(folder / 'summary.json', summary)
        status.pop('reason', None)
        status.update(state='completed', checkpoint='final_checkpoint.rrtl', checkpoint_sha256=file_hash(final))
        publish()
        return status
    except TimeoutError as exc:
        status.update(state='partial', reason=str(exc))
        publish()
        return status
    except Exception as exc:
        status.update(state='failed', reason=f'{type(exc).__name__}: {exc}')
        publish()
        raise
    finally:
        if session is not None:
            session.environment.close()


def _finals(output, manifest):
    for cell in manifest['cells']:
        folder = Path(output) / cell['id']
        status = _validate_run(manifest, cell, folder, complete=True)
        summary = read(folder / 'summary.json')
        path = folder / 'final_checkpoint.rrtl'
        if (status['interactions'] != manifest['protocol']['interactions'] or summary['state'] != 'completed'
                or summary['final_checkpoint_sha256'] != file_hash(path)):
            raise ValueError('Incomplete or altered final run')
        curve = []
        p = manifest['protocol']
        for point in p['curve_checkpoints']:
            saved = read(folder / f'validation_{point}.json')
            cp = folder / f'checkpoint_{point}.rrtl'
            validate_evaluation(saved, binding(manifest, cell, cp, 'validation', p['validation_seeds'], p['evaluation_horizon']))
            curve.append({'interactions': point, 'mean_return': saved['report']['mean_raw_return']})
        if summary['curve'] != curve or summary['validation_auc'] != learning_curve_area(curve):
            raise ValueError('Saved validation curve/AUC mismatch')
        yield cell, folder, path, summary


def evaluate_test(output, *, confirmed=False, evaluator=evaluate_saved):
    if not confirmed:
        raise ValueError('Test requires --confirm-test with the frozen completed protocol')
    manifest = load_frozen(output)
    p = manifest['protocol']
    finals = list(_finals(output, manifest))  # preflight ALL cells before any test seed
    pending = []
    for cell, folder, checkpoint, _ in finals:
        expected = binding(manifest, cell, checkpoint, 'test', p['test_seeds'], p['evaluation_horizon'])
        path = folder / 'test.json'
        if path.exists():
            validate_evaluation(read(path), expected)
        else:
            pending.append((cell, path, checkpoint, expected))
    for cell, path, checkpoint, expected in pending:
        result = evaluator(manifest, cell, checkpoint, 'test')
        validate_evaluation(result, expected)
        if path.exists():
            raise FileExistsError(path)
        write(path, result)


def aggregate(output, split='validation'):
    manifest = load_frozen(output, check_sources=False)
    p = manifest['protocol']
    values = {pipeline: {'bart': {}, 'corrected_common': {}} for pipeline in ('visual', 'ocatari')}
    runs = []
    for cell, folder, path, summary in _finals(output, manifest):
        filename = 'test.json' if split == 'test' else f"validation_{p['interactions']}.json"
        saved = read(folder / filename)
        validate_evaluation(saved, binding(manifest, cell, path, split, p[f'{split}_seeds'], p['evaluation_horizon']))
        values[cell['pipeline']][cell['learner']][cell['seed']] = saved['report']['mean_raw_return']
        rows = saved['report']['episodes']
        statistics = evaluation_summary(rows, p['evaluation_horizon'])
        runs.append({'median_raw_return': statistics['median_raw_return'],
                     'std_raw_return': statistics['std_raw_return'], 'iqr_raw_return': statistics['iqr_raw_return'],
                     'evaluation_points': ([{k: row[k] for k in ('seed', 'points_won', 'points_lost')} for row in rows]
                         if saved['report'].get('schema') == 'pong_evaluation_v2' else None),
                     'id': cell['id'], 'return': saved['report']['mean_raw_return'],
                     'truncation_fraction': saved['report']['truncation_fraction'],
                     'timing': summary['timing'], 'validation_auc': summary['validation_auc'],
                     'curve': summary['curve'], 'complexity': summary['metrics']['complexity']})
    limits, revisions = operational_budgets(output, manifest)
    report = {'split': split, 'protocol_hash': digest(manifest), 'runs': runs,
              'operational_budgets': limits, 'operational_budget_revision': len(revisions),
              'contrasts': {pipeline: paired_bootstrap(v['bart'], v['corrected_common']) for pipeline, v in values.items()},
              'reporting_sources': sources(), 'limitations': [
                  'Exploratory; bootstrap unit is training seed, not episode. One/three seeds are descriptive.',
                  'Compare learners within pipeline only; no causal claim about extraction.',
                  'Horizon-limited returns are not complete games. Short BART fits are approximate.']}
    write(Path(output) / f'report_{split}.json', report)
    return report


def campaign(output, *, raw=None, training_seconds=None, evaluation_seconds=None, budget_reason=None):
    if raw is not None and any(v is not None for v in (training_seconds, evaluation_seconds, budget_reason)):
        raise ValueError('Budget extensions are only allowed with resume')
    output = Path(output).resolve()
    manifest = freeze(raw, output) if raw is not None else load_frozen(output)
    if any(v is not None for v in (training_seconds, evaluation_seconds, budget_reason)):
        extend_budgets(output, manifest, training_seconds=training_seconds,
                       evaluation_seconds=evaluation_seconds, reason=budget_reason)
    limits, revisions = operational_budgets(output, manifest)
    budget = limits['evaluation_seconds_total']
    spent = sum(read(path).get('evaluation_seconds', 0.) for path in output.glob('*/*/*/status.json'))
    for cell_id in manifest['execution_order']:
        folder = output / cell_id
        cell = next(c for c in manifest['cells'] if c['id'] == cell_id)
        before = 0.
        if (folder / 'status.json').exists():
            status = _validate_run(manifest, cell, folder)
            if status['state'] == 'completed':
                continue
            before = status.get('evaluation_seconds', 0.)
        if spent >= budget:
            break
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'experiments.compare_pong', '--output', str(output),
                   '--worker', cell_id, '--evaluation-budget', str(budget - spent)]
        with (folder / 'worker.log').open('a') as log:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=limits['training_seconds_per_run'] + budget - spent + 120)
            except subprocess.TimeoutExpired:
                status = read(folder / 'status.json') if (folder / 'status.json').exists() else {
                    'cell_hash': cell['hash'], 'protocol_hash': digest(manifest)}
                status.update(state='partial', reason='worker_timeout_last_checkpoint_preserved')
                write(folder / 'status.json', status)
                break
        if (folder / 'status.json').exists():
            spent += read(folder / 'status.json').get('evaluation_seconds', 0.) - before
        if result.returncode:
            raise RuntimeError(f'Run failed: {cell_id}; see {folder / "worker.log"}')
    statuses = [read(output / c['id'] / 'status.json')['state'] if (output / c['id'] / 'status.json').exists()
                else 'not_run' for c in manifest['cells']]
    write(output / 'campaign_status.json', {'states': dict(zip([c['id'] for c in manifest['cells']], statuses)),
                                           'evaluation_seconds': spent, 'operational_budgets': limits,
                                           'operational_budget_revision': len(revisions)})
    if all(state == 'completed' for state in statuses):
        aggregate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage', choices=('train', 'resume', 'report-validation', 'test', 'report-test'), default='train')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--confirm-test', action='store_true')
    parser.add_argument('--cost-reference', type=Path,
                        help='Completed smoke campaign for a measured dry-run time estimate')
    parser.add_argument('--worker')
    parser.add_argument('--evaluation-budget', type=float)
    parser.add_argument('--training-budget-seconds', type=float, help='Resume: new cumulative training limit per run')
    parser.add_argument('--evaluation-budget-seconds', type=float, help='Resume: new cumulative campaign evaluation limit')
    parser.add_argument('--budget-reason', help='Required justification recorded with the budget extension')
    args = parser.parse_args()
    extension = any(v is not None for v in (args.training_budget_seconds, args.evaluation_budget_seconds, args.budget_reason))
    if extension and (args.stage != 'resume' or args.worker or args.dry_run):
        parser.error('Budget extension flags are only allowed with --stage resume')
    if args.dry_run:
        if not args.protocol:
            parser.error('--dry-run requires --protocol')
        resolved = resolve_protocol(read(args.protocol))
        p = resolved['protocol']
        estimate = None
        if args.cost_reference:
            from experiments.audit_pong_readiness import estimate_cost
            reference = load_frozen(args.cost_reference, check_sources=False)
            summaries = [(cell, summary, [read(folder / f'validation_{n}.json')
                         for n in reference['protocol']['curve_checkpoints']])
                         for cell, folder, _, summary in _finals(args.cost_reference, reference)]
            estimate = estimate_cost(reference, summaries, p)
        print(json.dumps({**resolved, 'measured_cost_estimate': estimate, 'cost_workload': {'runs': len(resolved['cells']),
            'training_decisions': len(resolved['cells']) * p['interactions'],
            'validation_episodes': len(resolved['cells']) * len(p['curve_checkpoints']) * p['evaluation_episodes'],
            'validation_decision_upper_bound': len(resolved['cells']) * len(p['curve_checkpoints']) * p['evaluation_episodes'] * p['evaluation_horizon'],
            'wall_time_estimate': 'Use audit_pong_readiness measured smoke report; no emulator or training in dry-run'}}, indent=2))
        return
    if args.output is None:
        parser.error('--output required')
    if args.worker:
        run_one(args.output, args.worker, evaluation_budget=args.evaluation_budget)
    elif args.stage == 'train':
        if not args.protocol:
            parser.error('train requires --protocol')
        campaign(args.output, raw=read(args.protocol))
    else:
        if args.protocol:
            parser.error('Use only the frozen protocol after train')
        if args.stage == 'resume':
            campaign(args.output, training_seconds=args.training_budget_seconds,
                     evaluation_seconds=args.evaluation_budget_seconds, budget_reason=args.budget_reason)
        elif args.stage == 'test':
            evaluate_test(args.output, confirmed=args.confirm_test)
            aggregate(args.output, 'test')
        else:
            aggregate(args.output, 'test' if args.stage == 'report-test' else 'validation')


if __name__ == '__main__':
    main()
