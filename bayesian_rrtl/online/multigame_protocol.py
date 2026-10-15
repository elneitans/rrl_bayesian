"""C15: explicit game/pipeline/learner/seed cells resolved by the B protocol engine."""
from dataclasses import asdict
import copy

from bayesian_rrtl.online.environments import GameEnvironmentConfig
from experiments.pong_analysis_runtime import digest

SCHEMA='atari_comparison_v3'


def resolve(raw):
    from .protocol import resolve as resolve_b
    p=copy.deepcopy(raw)
    required={'schema','name','level','games','learners','learning','exploration','cache_bytes',
              'bart_review','primary_checkpoint','cells','resources'}
    if set(p) != required or p['schema'] != SCHEMA or not p['games'] or not p['cells']:
        raise ValueError('Invalid multigame protocol fields/cells')
    if p['primary_checkpoint'] != 'final' or p['level'] not in ('smoke','baseline','long'):
        raise ValueError('Final primary checkpoint and explicit execution level required')
    if set(p['resources']) != {'cpus_per_cell','memory_gib_per_cell'} or any(
            type(v) is not int or v < 1 for v in p['resources'].values()):
        raise ValueError('Explicit CPU/memory resources required')
    resolved_games={}
    unresolved=[]
    game_fields={'environment','actions','relations','training_seeds','evaluation','max_decisions',
                 'milestones','checkpoint_interval','budgets','auc_interval','contrasts','primary_metric'}
    for game,g in p['games'].items():
        if set(g) != game_fields or g['primary_metric'] != 'final_development_mean_raw_return':
            raise ValueError('Explicit per-game conditions, metric, AUC and contrasts required')
        env=GameEnvironmentConfig(**g['environment'])
        if env.game != game or g['actions'] != list(env.spec().actions) or digest(g['relations']) != digest(env.relations().manifest()):
            raise ValueError('Declared game actions/relations differ from audited contract')
        maximum=g['max_decisions']
        interval=g['auc_interval']
        points=g['evaluation']['intermediate']['checkpoints']
        if maximum is not None and (not interval or len(interval) != 2 or interval[0] >= interval[1]
                or interval[0] not in points or interval[1] not in points):
            raise ValueError('AUC interval must be predeclared within intermediate calendar')
        if maximum is None:
            unresolved.append(game+':auc_interval/calendar')
        for contrast in g['contrasts']:
            if len(contrast) != 2 or any(n not in p['learners'] for n in contrast) or contrast[0] == contrast[1]:
                raise ValueError('Unknown within-game learner contrast')
        # Reuse B validation and scientific schedule, with an explicit action/environment replacement.
        b={'schema':'pong_progressive_v2','name':p['name']+':'+game,'level':p['level'],
           'pipelines':['ocatari'],'rom_sha256':env.rom_sha256,'training_horizon':env.horizon,
           'learners':p['learners'],'learning':p['learning'],'exploration':p['exploration'],
           'training_seeds':g['training_seeds'],'evaluation':g['evaluation'],'max_decisions':maximum,
           'milestones':g['milestones'],'checkpoint_interval':g['checkpoint_interval'],'budgets':g['budgets'],
           'cache_bytes':p['cache_bytes'],'bart_review':p['bart_review'],'primary_checkpoint':'final'}
        resolved_games[game]=resolve_b(b)
        unresolved += [game+':'+v for v in resolved_games[game]['unresolved']]
    cells=[]
    seen=set()
    for selection in p['cells']:
        if set(selection) != {'game','pipeline','learner','seed'}:
            raise ValueError('Each cell explicitly declares game/pipeline/learner/seed')
        game,pipeline,learner,seed=(selection[k] for k in ('game','pipeline','learner','seed'))
        if game not in resolved_games:
            raise ValueError('Cell refers to undeclared game')
        g=p['games'][game]
        if learner not in p['learners'] or seed not in g['training_seeds']:
            raise ValueError('Cell learner/seed not predeclared')
        env=GameEnvironmentConfig(**{**g['environment'],'pipeline':pipeline})
        base=next(c for c in resolved_games[game]['cells'] if c['learner'] == learner and c['seed'] == seed)
        cell={**{k:v for k,v in base.items() if k not in ('hash','id','environment')},
              'game':game,'pipeline':pipeline,'environment':asdict(env),
              'id':f'{game}/{pipeline}/{learner}/{seed}',
              'learning':{**base['learning'],'actions':env.spec().actions}}
        if cell['id'] in seen:
            raise ValueError('Duplicate explicit cell')
        seen.add(cell['id'])
        cells.append({**cell,'hash':digest(cell)})
    for game,g in p['games'].items():
        if not any(c['game'] == game for c in cells):
            raise ValueError('Declared game has no reviewed cells')
        for pipeline in {c['pipeline'] for c in cells if c['game'] == game}:
            methods={c['learner'] for c in cells if c['game'] == game and c['pipeline'] == pipeline}
            for a,b in g['contrasts']:
                if not {a,b} <= methods:
                    raise ValueError('Contrast lacks explicit cells')
    # Shared B helpers consume this projection; every heterogeneous field lives in games.
    projection={**p,'max_decisions':None,'milestones':[], 'evaluation':{},'budgets':{},'checkpoint_interval':None}
    return {'schema':SCHEMA,'protocol':projection,'cells':cells,'unresolved':sorted(set(unresolved))}


def cell_protocol(manifest, cell):
    p=manifest['protocol']
    if manifest['schema'] != SCHEMA:
        return p
    return {**p,**p['games'][cell['game']]}


def raw_protocol(manifest):
    p=copy.deepcopy(manifest['protocol'])
    if manifest['schema'] == SCHEMA:
        for name in ('max_decisions','milestones','evaluation','budgets','checkpoint_interval'):
            p.pop(name,None)
    return p


def dry_run(raw, measurements=None):
    from .protocol import dry_run as dry_b, estimate_cell_cost
    from .protocol import resolve as resolve_b
    resolved=resolve(raw)
    p=resolved['protocol']
    workload=[]
    estimates=[]
    for cell in resolved['cells']:
        g=p['games'][cell['game']]
        one={'schema':'pong_progressive_v2','name':p['name'],'level':p['level'],
            'pipelines':['ocatari'],'rom_sha256':cell['environment']['rom_sha256'],
            'training_horizon':cell['environment']['horizon'],'learners':p['learners'],
            'learning':p['learning'],'exploration':p['exploration'],'training_seeds':g['training_seeds'],
            'evaluation':g['evaluation'],'max_decisions':g['max_decisions'],'milestones':g['milestones'],
            'checkpoint_interval':g['checkpoint_interval'],'budgets':g['budgets'],
            'cache_bytes':p['cache_bytes'],'bart_review':p['bart_review'],'primary_checkpoint':'final'}
        _=resolve_b(one)
        d=dry_b(one,None)
        row=next(r for r in d['workload'] if r['id'] == f"ocatari/{cell['learner']}/{cell['seed']}")
        workload.append({**row,'id':cell['id'],'game':cell['game'],'pipeline':cell['pipeline'],
                         'resources':p['resources']})
        estimate=estimate_cell_cost(cell,{**p,**g},measurements,g['max_decisions'],row['expected_checkpoints'])
        if estimate is not None:
            estimates.append(estimate)
    return {'resolved':resolved,'workload':workload,'estimates':estimates,
            'n_trainings':len(workload),'n_development_evaluation_episodes':sum(r['evaluation_episodes'] for r in workload),
            'pending_authorizations':resolved['unresolved']+([] if p['level'] == 'smoke' else ['explicit game milestones']),
            'uncertainty':'Pong B measurements cannot estimate Breakout/DemonAttack. Missing compatible Linux cost is unknown.',
            'reserved_test_episodes_executed':0}
