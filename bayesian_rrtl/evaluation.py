"""Seed-level reports and paired bootstrap; episodes are not independent runs."""

import numpy as np


def seed_summary(values, *, failure_threshold):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Need finite returns per training seed")
    return {
        "n_training_seeds": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "failure_threshold": failure_threshold,
        "failure_fraction": float(np.mean(values < failure_threshold)),
    }


def paired_bootstrap(left, right, *, seed=912345, draws=5000):
    if set(left) != set(right) or not left:
        raise ValueError("Paired comparisons need the same nonempty training seeds")
    keys = sorted(left)
    differences = np.array([left[k] - right[k] for k in keys], dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Non-finite differences")
    bootstrap = (
        np.random.default_rng(seed)
        .choice(differences, (draws, len(keys)), replace=True)
        .mean(axis=1)
    )
    return {
        "training_seeds": keys,
        "mean_difference": float(differences.mean()),
        "bootstrap_95_percentile": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "interpretation": "Exploratory seed-level interval; no multiplicity-adjusted confirmatory claim.",
    }


def learning_curve_area(points):
    points = sorted(points, key=lambda p: p["interactions"])
    x = np.array([p["interactions"] for p in points], dtype=float)
    y = np.array([p["mean_return"] for p in points], dtype=float)
    if len(x) < 2 or np.any(np.diff(x) <= 0) or not np.isfinite(y).all():
        raise ValueError("Learning curve needs distinct increasing checkpoints")
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def export_rules(model, batch, *, bayesian=True):
    """Posterior inclusion frequency by action; no claim of one faithful program."""
    from dataclasses import asdict
    import json

    output = {}
    for action, posterior in zip(model.actions, model.posteriors):
        if posterior is None:
            output[action] = {"fitted": False}
            continue
        draws = [d.trees for d in posterior.draws] if bayesian else [(posterior.tree,)]
        counts = {}
        contributions = []
        for trees in draws:
            seen = set()
            for tree in trees:
                for _, node, _ in tree.walk(batch):
                    if not node.is_leaf:
                        seen.add(json.dumps(asdict(node.rule), sort_keys=True))
            for rule in seen:
                counts[rule] = counts.get(rule, 0) + 1
        trees = draws[-1]
        for index, tree in enumerate(trees):
            contributions.append(
                {"tree": index, "reference_contribution": tree.predict(batch).tolist()}
            )
        output[action] = {
            "fitted": True,
            "frequency_kind": "posterior inclusion"
            if bayesian
            else "point model inclusion",
            "rules": [
                {"rule": json.loads(rule), "frequency": count / len(draws)}
                for rule, count in sorted(counts.items())
            ],
            "last_draw_contributions": contributions,
            "units": "normalized; add intercept and multiply scale once per ensemble"
            if bayesian
            else "original reward units",
        }
    return output


def comparative_report(manifest, rows):
    """C15: independent units are training seeds; never average raw scores across games."""
    from .online.multigame_protocol import cell_protocol
    by_id={c['id']:c for c in manifest['cells']}
    groups={}
    inventory=[]
    for row in rows:
        cell=by_id[row['cell']]
        p=cell_protocol(manifest,cell)
        game,pipeline,method,seed=cell['game'],cell['pipeline'],cell['learner'],cell['seed']
        curve=sorted([{'interactions':e['counter'],'mean_return':e['report']['mean_raw_return'],
                       'training_active_seconds':e.get('training_active_seconds'),
                       'completed_games':e['report']['completed_games'],
                       'truncation_fraction':e['report']['truncation_fraction']}
                      for e in row.get('evaluations',[]) if e['group']=='intermediate'],
                     key=lambda c:c['interactions'])
        interval=p['auc_interval']
        expected=[n for n in p['evaluation']['intermediate']['checkpoints'] if interval and interval[0]<=n<=interval[1]]
        selected=[v for v in curve if interval and interval[0]<=v['interactions']<=interval[1]]
        auc=learning_curve_area(selected) if len(selected)>=2 and [v['interactions'] for v in selected]==expected else None
        final=next((e for e in row.get('evaluations',[]) if e['group']=='final_development'
                    and e['counter']==p['max_decisions']),None)
        result={'cell':row['cell'],'game':game,'pipeline':pipeline,'learner':method,'seed':seed,
                'state':row['comparison_state'],'counter':row.get('counter',0),
                'maximum_decisions':p['max_decisions'],'curve':curve,'auc_interval':interval,
                'normalized_auc_decisions':auc,'auc_status':'complete' if auc is not None else 'inconclusive_missing_common_interval',
                'training_seconds':row.get('training_seconds') if row.get('training_seconds_complete',True) else None,
                'training_seconds_recorded_lower_bound':row.get('training_seconds'),
                'unfinished_training_attempts':row.get('unfinished_training_attempts',0),
                'evaluation_seconds':row.get('evaluation_seconds') if row.get('evaluation_seconds_complete',True) else None,
                'metrics':row.get('metrics'),'retries':max(0,row.get('training_attempts',0)-1),
                'peak_rss_bytes':row.get('peak_rss_bytes'),
                'queue_seconds':None,'final_development':final['report'] if final else None}
        inventory.append(result)
        group=groups.setdefault((game,pipeline,method),{'returns':{},'aucs':{},'scopes':{}})
        if final and row['training_state']=='completed':
            group['returns'][seed]=final['report']['mean_raw_return']
            ep=final['report']['episodes']
            group['scopes'][seed]={'completed_games':sum(bool(e['terminated']) for e in ep),
                'truncation_fraction':float(np.mean([e['truncated'] for e in ep])),
                'mean_by_scope':{scope:float(np.mean([e['return'] for e in ep if e['return_scope']==scope]))
                    for scope in sorted({e['return_scope'] for e in ep})}}
        if auc is not None:
            group['aucs'][seed]=auc
    summary=[]
    for (game,pipeline,method),group in sorted(groups.items()):
        values=list(group['returns'].values())
        summary.append({'game':game,'pipeline':pipeline,'learner':method,
            'independent_unit':'training_seed','per_training_seed':group['returns'],
            'return_scope_composition':group['scopes'],
            'n_training_seeds':len(values),'mean':float(np.mean(values)) if values else None,
            'median':float(np.median(values)) if values else None,
            'std':float(np.std(values,ddof=1)) if len(values)>1 else None,
            'iqr':float(np.percentile(values,75)-np.percentile(values,25)) if values else None})
    contrasts=[]
    for game,g in manifest['protocol']['games'].items():
        pipelines=sorted({c['pipeline'] for c in manifest['cells'] if c['game']==game})
        for pipeline in pipelines:
            for left,right in g['contrasts']:
                a=groups.get((game,pipeline,left),{'returns':{},'aucs':{}})
                b=groups.get((game,pipeline,right),{'returns':{},'aucs':{}})
                for metric,key in (('final_development_raw_return','returns'),('auc_decisions','aucs')):
                    common=sorted(set(a[key]) & set(b[key]))
                    record={'game':game,'pipeline':pipeline,'left':left,'right':right,'metric':metric,
                            'matched_training_seeds':common,
                            'missing_left':sorted(set(b[key])-set(a[key])),
                            'missing_right':sorted(set(a[key])-set(b[key])),
                            'state':'exploratory' if common else 'inconclusive'}
                    if common:
                        record['bootstrap']=paired_bootstrap({s:a[key][s] for s in common},{s:b[key][s] for s in common})
                    contrasts.append(record)
    return {'runs':inventory,'seed_summaries':summary,'paired_contrasts':contrasts,
            'no_cross_game_score_average':True,
            'limits':['Few training seeds and degenerate bootstrap intervals do not establish equivalence.',
                      'Fixed-data A diagnostics do not certify every online BART fit.',
                      'A poor policy alone does not establish overfit, bad mixing or representation insufficiency.',
                      'Active times exclude queue and process startup; sums are not parallel campaign wall time.']}


def export_interpretability(agent, learner, references):
    """Faithful model components and per-action Q, not a surrogate decision program."""
    from dataclasses import asdict
    from .online.learners import LEARNERS
    if learner not in LEARNERS:
        raise ValueError('Unknown learner')
    batch=agent.encoder.transform(references)
    policy='argmax over all action Q; uniform RNG tie breaking'
    if learner=='corrected_common':
        output={}
        def visit(node,path):
            entry={'conditions':path,'q':node.q_value,'internal_fallback':node.q_value}
            if node.literal is None:
                return [entry]
            rows=[{**entry,'split_literal':dict(node.literal._asdict()),'missing':'return internal node Q'}]
            for label,child in zip(('more','same','less') if node.literal.type=='comparative' else ('True','False'),node.children):
                rows+=visit(child,path+[{'literal':dict(node.literal._asdict()),'value':label}])
            return rows
        for name,root in agent.root_nodes.items():
            output[name]=visit(root,[])
        q=np.array([[agent.root_nodes[a].predict(agent._state(s)) for a in agent.actions] for s in references])
        return {'learner':learner,'policy':policy,'actions':output,'reference_q':q.tolist(),
                'units':'training clipped reward Q','warning':'One action-tree path is not the complete decision policy.'}
    q=agent.model.predict(batch)
    output=export_rules(agent.model,batch,bayesian=LEARNERS[learner].posterior) if learner!='extra_trees' else {}
    if learner=='cart':
        for name,model in zip(agent.config.actions,agent.model.posteriors):
            if model is None:
                continue
            paths=[]
            def walk(tree,conditions):
                if tree.is_leaf:
                    paths.append({'conditions':conditions,'q':tree.mu})
                else:
                    for branch,child in ((True,tree.true_child),(False,tree.false_child)):
                        walk(child,conditions+[{'rule':asdict(tree.rule),'branch':branch}])
            walk(model.tree,[])
            output[name]['paths']=paths
            output[name]['missing']='equals on missing routes false; is_missing can split explicitly'
    result={'learner':learner,'policy':policy,'actions':output,'reference_q':q.tolist(),
            'reference_values':batch.values.tolist(),'reference_observed':batch.observed.tolist(),
            'catalog':agent.encoder.to_dict(),'units':'training clipped reward Q',
            'warning':'Frequent splits are not an equivalent policy; prediction combines all trees and posterior draws.'}
    if learner=='bart':
        result['normalization']={'intercept_c':agent.config.scale.c,'width_W':agent.config.scale.width,
            'recover':'Q = c + W * posterior_mean(sum(tree_contributions)); no observation noise in greedy Q'}
    if learner=='extra_trees':
        result['warning']='ExtraTrees uses nominal/missing one-hot features; this export provides exact Q, not relational surrogate rules.'
    return result
