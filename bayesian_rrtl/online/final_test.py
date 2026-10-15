"""C15 explicit final-test seal. Ordinary evaluation/report never enters this module."""
import json
from pathlib import Path

from .protocol import load
from .multigame_protocol import cell_protocol
from .storage import binding,load_frontier
from experiments.pong_analysis_runtime import (checked_read,checked_write,bytes_hash,digest,
                                               episode,unit_lock)


def final_packet(output):
    from experiments.compare_pong_progressive import consolidate,validate_status
    output=Path(output)
    manifest=load(output)
    if (output/'final_test'/'usage.json').exists():
        raise ValueError('Test has been opened; selection cannot be prepared again')
    if manifest['schema'] != 'atari_comparison_v3' or manifest['unresolved']:
        raise ValueError('Final test requires resolved multigame protocol')
    development=consolidate(output)
    if not development['completed']:
        raise ValueError('All training and development evaluation must be complete')
    finals=[]
    for cell in manifest['cells']:
        p=cell_protocol(manifest,cell)
        folder=output/cell['id']
        status=validate_status(checked_read(folder/'status.json'),manifest,cell)
        if status['counter'] != p['max_decisions'] or status['state'] != 'completed':
            raise ValueError('Final checkpoint is incomplete')
        load_frontier(folder,status,binding(manifest,cell))
        finals.append({'cell':cell['id'],'checkpoint':status['checkpoint'],
                       'counter':status['counter'],'seeds':p['evaluation']['reserved_test']['seeds'],
                       'horizon':p['evaluation']['reserved_test']['horizon'],'policy':'greedy_uniform_ties_v1'})
    return {'schema':'atari_final_selection_v1','campaign':digest(manifest),'finals':finals,
            'development_report_sha256':digest(development),
            'selection_decisions':None,'test_seeds_untouched_attestation':False}


def prepare(output, path):
    packet=final_packet(output)
    from experiments.pong_analysis_runtime import atomic_json
    atomic_json(path,packet,replace=False)
    return packet


def authorize(output, selection, *, confirmed=False, reason=None):
    if not confirmed or not isinstance(reason,str) or not reason.strip():
        raise ValueError('Explicit --confirm-test and selection review reason required')
    packet=json.loads(Path(selection).read_text())
    sealed=Path(output)/'final_test'/'authorization.json'
    seal={'packet':packet,'selection_sha256':bytes_hash(Path(selection).read_bytes()),'reason':reason}
    if sealed.exists():
        previous=checked_read(sealed)
        if previous != seal:
            raise ValueError('Final selection is sealed; test cannot be reused for retuning')
        load(output)
        return previous
    expected=final_packet(output)
    if set(packet) != set(expected):
        raise ValueError('Final selection packet fields mismatch')
    immutable={k:v for k,v in packet.items() if k not in ('selection_decisions','test_seeds_untouched_attestation')}
    if digest(immutable) != digest({k:v for k,v in expected.items() if k in immutable}):
        raise ValueError('Test selection packet differs from frozen final checkpoints/development')
    if not packet['selection_decisions'] or packet['test_seeds_untouched_attestation'] is not True:
        raise ValueError('Record development-only selection and attest untouched test seeds')
    seal={'packet':packet,'selection_sha256':bytes_hash(Path(selection).read_bytes()),'reason':reason}
    folder=Path(output)/'final_test'
    with unit_lock(folder):
        path=folder/'authorization.json'
        if path.exists():
            if checked_read(path) != seal:
                raise ValueError('Final selection is sealed; test cannot be reused for retuning')
        else:
            checked_write(path,seal,replace=False)
    return seal


def evaluate(output, selection, *, confirmed=False, factory=None, extend_seconds=0., extension_id=None, reason=None):
    from .protocol import select_cell
    from experiments.compare_pong_progressive import default_factory,GreedyController,validate_extension
    if not confirmed:
        raise ValueError('Reserved evaluation requires --confirm-test on every invocation')
    validate_extension(extend_seconds,extension_id,reason)
    output=Path(output)
    manifest=load(output)
    cell=select_cell(manifest,selection)
    seal=checked_read(output/'final_test'/'authorization.json')
    if seal['packet']['campaign'] != digest(manifest):
        raise ValueError('Sealed test campaign differs')
    final=next(v for v in seal['packet']['finals'] if v['cell']==cell['id'])
    folder=output/cell['id']
    status=checked_read(folder/'status.json')
    if status['checkpoint'] != final['checkpoint'] or status['counter'] != final['counter']:
        raise ValueError('Final checkpoint changed after selection')
    payload,_=load_frontier(folder,status,binding(manifest,cell))
    p=cell_protocol(manifest,cell)
    test_factory=(factory or default_factory(cell)).with_horizon(final['horizon'])
    rows=[]
    # Publishing usage before opening any test environment closes the tuning boundary.
    with unit_lock(output/'final_test'):
        usage=output/'final_test'/'usage.json'
        used=checked_read(usage) if usage.exists() else {'selection':digest(seal),'cells_started':[]}
        if used['selection'] != digest(seal):
            raise ValueError('Reserved seeds already bound to another selection')
        if cell['id'] not in used['cells_started']:
            used['cells_started'].append(cell['id'])
            checked_write(usage,used)
    for seed in final['seeds']:
        identity={'schema':'atari_reserved_test_v1','selection':digest(seal),'campaign':digest(manifest),
                  'cell':cell['hash'],'checkpoint':final['checkpoint'],'seed':seed,
                  'seeds':final['seeds'],'horizon':final['horizon'],'policy':final['policy']}
        unit=folder/'evaluations'/'reserved_test'/str(final['counter'])/str(seed)
        if extend_seconds:
            with unit_lock(unit):
                path=unit/'budget_reviews.json'
                reviews=checked_read(path) if path.exists() else []
                review={'id':extension_id,'seconds':extend_seconds,'reason':reason}
                old=next((r for r in reviews if r['id']==extension_id),None)
                if old and old != review:
                    raise ValueError('Test budget review changed')
                if old is None:
                    checked_write(path,reviews+[review])
        row=episode(test_factory,GreedyController(payload['session']['collector']['agent'],cell['learner']),
                    unit,identity,
                    seed=seed,horizon=final['horizon'],seconds=p['budgets']['evaluation_seconds_per_episode'],
                    rng_seed=seed,pong_points=cell['game']=='Pong',extend_seconds=extend_seconds,extension_id=extension_id)
        rows.append(row)
    return {'cell':cell['id'],'checkpoint':final['checkpoint'],'episodes':rows,
            'state':'completed' if all(r['status']=='completed' for r in rows) else 'partial'}


def report(output):
    """Read-only reserved evaluation report: pending episodes stay pending, never run."""
    from .protocol import select_cell
    from bayesian_rrtl.comparison import evaluation_summary
    from bayesian_rrtl.evaluation import paired_bootstrap
    from experiments.pong_analysis_runtime import WorkUnit
    output=Path(output)
    manifest=load(output)
    seal=checked_read(output/'final_test'/'authorization.json')
    if seal['packet']['campaign']!=digest(manifest):
        raise ValueError('Test seal/campaign mismatch')
    rows=[]
    scores={}
    for final in seal['packet']['finals']:
        cell=select_cell(manifest,final['cell'])
        episodes=[]
        pending=[]
        pending_details=[]
        for seed in final['seeds']:
            folder=output/cell['id']/'evaluations'/'reserved_test'/str(final['counter'])/str(seed)
            if not (folder/'identity.json').exists():
                pending.append(seed)
                pending_details.append({'seed':seed,'reason':'evaluation_pending'})
                continue
            expected={'schema':'atari_reserved_test_v1','selection':digest(seal),'campaign':digest(manifest),
                      'cell':cell['hash'],'checkpoint':final['checkpoint'],'seed':seed,
                      'seeds':final['seeds'],'horizon':final['horizon'],'policy':final['policy']}
            try:
                with unit_lock(folder):
                    unit=WorkUnit(folder,expected)
                    saved=unit.load()
            except RuntimeError as exc:
                if not isinstance(exc.__cause__,BlockingIOError):
                    raise
                # The evaluator may be replacing/retiring progress blobs. Do
                # not inspect its frontier until the episode lock is released.
                pending.append(seed)
                pending_details.append({'seed':seed,'reason':'evaluation_running'})
                continue
            if saved is None or saved['status']!='completed':
                pending.append(seed)
                pending_details.append({'seed':seed,'reason':'evaluation_partial'})
            else:
                ep=saved['result']
                episodes.append({**ep,'return':ep['raw_return']})
        statistics=evaluation_summary(episodes,final['horizon']) if not pending else None
        rows.append({'cell':cell['id'],'game':cell['game'],'pipeline':cell['pipeline'],'learner':cell['learner'],
                     'training_seed':cell['seed'],'checkpoint':final['checkpoint'],
                     'state':'completed' if not pending else 'partial','pending_evaluation_seeds':pending,
                     'pending_evaluations':pending_details,'report':statistics})
        if statistics:
            scores.setdefault((cell['game'],cell['pipeline'],cell['learner']),{})[cell['seed']]=statistics['mean_raw_return']
    contrasts=[]
    for game,g in manifest['protocol']['games'].items():
        for pipeline in sorted({c['pipeline'] for c in manifest['cells'] if c['game']==game}):
            for a,b in g['contrasts']:
                left=scores.get((game,pipeline,a),{})
                right=scores.get((game,pipeline,b),{})
                common=sorted(set(left)&set(right))
                contrasts.append({'game':game,'pipeline':pipeline,'left':a,'right':b,'matched_training_seeds':common,
                                  'bootstrap':paired_bootstrap({s:left[s] for s in common},{s:right[s] for s in common}) if common else None})
    value={'split':'reserved_test','selection':digest(seal),'campaign':digest(manifest),
           'state':'completed' if all(r['state']=='completed' for r in rows) else 'partial',
           'runs':rows,'paired_contrasts':contrasts,'independent_unit':'training_seed',
           'limits':['Test observed: no retuning with these reserved seeds.',
                     'Few seeds/degenerate intervals do not establish equivalence.']}
    checked_write(output/'final_test'/'report.json',value)
    return value
