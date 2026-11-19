"""Summarize downloaded Breakout metadata; does not load model/log blobs.

Standalone stdlib utility outside the frozen experiments/model source set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

METHODS = ('corrected_common', 'cart', 'extra_trees', 'bart')
SEEDS = tuple(range(710601, 710611))
POINT = 1_000_000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    data = json.loads(path.read_text())
    if data.get('schema') != 'pong_analysis_unit_v2' or digest(data['payload']) != data['sha256']:
        raise ValueError(f'Invalid hash/schema: {path}')
    return data['payload']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(root):
    rows = []
    shared = None
    for method in METHODS:
        folder = root / method
        manifest = read(folder / 'campaign.json')
        campaign = digest(manifest)
        p = manifest['protocol']
        game = p['games']['Breakout']
        evaluation = game['evaluation']['final_development']
        require(game['training_seeds'] == list(SEEDS), 'Unexpected training seeds')
        require(game['max_decisions'] == 2_000_000 and POINT in game['milestones'],
                'Missing 1M milestone or future 2M maximum')
        require(POINT in evaluation['checkpoints'] and len(evaluation['seeds']) == 10,
                'Expected ten development episodes at 1M')
        require(not game['evaluation']['reserved_test']['checkpoints'], 'Reserved test scheduled')
        expected_ids = {f'Breakout/ocatari/{method}/{seed}' for seed in SEEDS}
        require(len(manifest['cells']) == 10 and {c['id'] for c in manifest['cells']} == expected_ids,
                'Unexpected, missing or duplicate cells')
        signature = {'environment': game['environment'], 'relations': game['relations'],
                     'actions': game['actions'], 'learning': p['learning'],
                     'exploration': p['exploration'], 'evaluation': game['evaluation'],
                     'source_hashes': manifest['source_hashes']}
        if shared is None:
            shared = signature
        require(signature == shared, 'Conditions have incompatible shared contracts')
        for cell in sorted(manifest['cells'], key=lambda c: c['id']):
            seed = cell['learning']['seed']
            require(cell['id'] == f'Breakout/ocatari/{method}/{seed}' and cell['learner'] == method,
                    'Cell seed/learner mismatch')
            cell_folder = folder / cell['id']
            status_file = cell_folder / 'status.json'
            row = {'method': method, 'seed': seed, 'counter': 0, 'evaluation_episodes': 0,
                   'mean_raw_return': None, 'truncated_episodes': 0, 'metadata_complete_1m': False}
            if not status_file.exists():
                rows.append(row)
                continue
            status = read(status_file)
            require(status['binding']['campaign'] == campaign and
                    status['binding']['cell'] == cell['hash'] and
                    status['binding']['learner'] == method, 'Status binding mismatch')
            row.update(counter=status['counter'], training_state=status['state'], reason=status['reason'])
            episodes = []
            for eval_seed in evaluation['seeds']:
                unit = cell_folder / 'evaluations' / 'final_development' / str(POINT) / str(eval_seed)
                if not (unit / 'state.json').exists():
                    continue
                identity = read(unit / 'identity.json')
                state = read(unit / 'state.json')
                require(state['identity_sha256'] == digest(identity), 'Evaluation identity hash mismatch')
                require(identity['campaign'] == campaign and identity['cell'] == cell['hash'] and
                        identity['seed'] == eval_seed and identity['seeds'] == evaluation['seeds'] and
                        identity['group'] == 'final_development' and
                        identity['horizon'] == evaluation['horizon'] and
                        identity['policy'] == 'greedy_uniform_ties_v1' and
                        identity['runtime'] == status['binding']['runtime'], 'Evaluation binding mismatch')
                ref = identity['checkpoint']
                require(ref['path'] == f'checkpoints/{POINT}-{ref["sha256"]}.pkl',
                        'Evaluation is not bound to the 1M checkpoint')
                if status['counter'] == POINT:
                    require(ref == status['checkpoint'], 'Status/evaluation checkpoint mismatch')
                if state['status'] != 'completed':
                    continue
                result = state['result']
                require(result['seed'] == eval_seed and result['status'] == 'completed' and
                        0 < result['steps'] <= evaluation['horizon'], 'Invalid episode result')
                episodes.append(result)
            row['evaluation_episodes'] = len(episodes)
            row['truncated_episodes'] = sum(e['truncated'] for e in episodes)
            if len(episodes) == 10:
                row['mean_raw_return'] = statistics.mean(e['raw_return'] for e in episodes)
                row['metadata_complete_1m'] = status['counter'] >= POINT
            rows.append(row)
    return {'scope': 'Downloaded metadata; checkpoint bytes and full log chains NOT verified here',
            'expected_models': 40, 'complete_models_metadata': sum(r['metadata_complete_1m'] for r in rows),
            'rows': rows}


def markdown(summary):
    lines = ['# Breakout a 1M: resumen de metadatos descargados', '',
             f'Modelos con 1M y diez evaluaciones completas: **{summary["complete_models_metadata"]}/40**.', '',
             'Este resumen valida hashes, identidades y contratos compartidos. No verifica los blobs ni '
             'las cadenas completas de logs; los reportes en compute siguen siendo necesarios.', '',
             '| Semilla | Baseline | CART | ExtraTrees | BART |',
             '|---|---:|---:|---:|---:|']
    index = {(r['method'], r['seed']): r for r in summary['rows']}
    for seed in SEEDS:
        values = []
        for method in METHODS:
            row = index[method, seed]
            value = f'{row["mean_raw_return"]:g}' if row['metadata_complete_1m'] else 'pendiente'
            if row['truncated_episodes']:
                value += f' ({row["truncated_episodes"]}/10 trunc.)'
            values.append(value)
        lines.append(f'| {seed} | ' + ' | '.join(values) + ' |')
    lines += ['', 'La unidad de comparación es la semilla de entrenamiento. Diez episodios por modelo '
              'no equivalen a diez agentes independientes. Los retornos truncados corresponden al '
              'horizonte declarado. No extraer una clasificación final de subconjuntos incompletos.', '',
              '| Método | Modelos evaluados | Media entre modelos disponibles | Mediana |',
              '|---|---:|---:|---:|']
    for method in METHODS:
        values = [r['mean_raw_return'] for r in summary['rows']
                  if r['method'] == method and r['metadata_complete_1m']]
        mean = f'{statistics.mean(values):.3f}' if values else '—'
        median = f'{statistics.median(values):.3f}' if values else '—'
        lines.append(f'| {method} | {len(values)}/10 | {mean} | {median} |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Markdown report; JSON saved alongside')
    args = parser.parse_args()
    summary = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(summary))
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'{summary["complete_models_metadata"]}/40 complete in metadata; report={args.output}')


if __name__ == '__main__':
    main()
