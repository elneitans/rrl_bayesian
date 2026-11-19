"""Independent read-only reconstruction of accepted repair losses and policy TV."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from bayesian_rrtl.targets import bootstrap_mask
from bayesian_rrtl.tree_revision import route,walk
from experiments.diagnose_rrtl_improvements import load_campaign
from experiments.diagnose_rrtl_revision import blob_read
from experiments.pong_analysis_runtime import checked_read,checked_write,digest


def audit(root,output):
    m,_=load_campaign(root);report=checked_read(root/'report.json')
    if not report['complete'] or report['missing'] or report['campaign']!=digest(m):
        raise ValueError('Complete report required')
    records=[]
    for cell in report['continuations']:
        folder=root/'continuations'/cell['criterion']/str(cell['seed'])/cell['mode']
        status=checked_read(folder/'status.json')
        for path in sorted((folder/'reviews').glob('*.json'),key=lambda p:int(p.stem)):
            if path.stem not in status['checkpoints']:continue
            saved=checked_read(path);accepted=[e for e in saved['reviews'] if e['gate']['accepted']]
            if not accepted:continue
            if len(accepted)!=1 or saved['identity']!=status['identity']:raise ValueError('Review binding mismatch')
            e=accepted[0];reference=blob_read(folder,saved['reference'])
            checkpoint=blob_read(folder,status['checkpoints'][path.stem]);agent=checkpoint['session']['collector']['agent']
            if reference['counter']!=checkpoint['counter']:raise ValueError('Pre/post counter mismatch')
            before,after=reference['roots'],agent.root_nodes;actions=agent.actions
            def signature(roots):
                return [(a,p,n.literal,n.q_value,n.revision_uid,n.refinements_stats)
                        for a,root_node in roots.items() for p,n in walk(root_node)]
            if signature(reference['target_roots'])!=signature(agent.target_roots):
                raise ValueError('Maintenance changed the frozen target model')
            if signature({a:v for a,v in before.items() if a!=e['action']})!=signature({a:v for a,v in after.items() if a!=e['action']}):
                raise ValueError('Maintenance changed an unrelated action tree')
            probes=blob_read(root/'holdouts',saved['holdout'])
            q=lambda roots,s:np.array([roots[a].predict(s) for a in actions])
            old_loss=[];new_loss=[];diffs={};tvs={}
            for row in probes:
                old,new=q(before,row['state']),q(after,row['state'])
                p,r=old==old.max(),new==new.max()
                tvs.setdefault(row['block'],[]).append(float(np.abs(p/p.sum()-r/r.sum()).sum()/2))
                if actions[row['action']]!=e['action'] or not any(n.revision_uid==e['node'] for _,n in route(before[e['action']],row['state'])):
                    continue
                mask=bootstrap_mask(row['terminated'],row['truncated'],
                    bootstrap_on_truncation=agent.config.bootstrap_on_truncation,
                    final_observation_available=row['next_state'] is not None)
                target=row['reward']
                if mask:
                    values=q(before,row['next_state'])
                    if reference['improvement']['bootstrap']=='double':
                        selected=np.flatnonzero(values==values.max())
                        bootstrap=float(q(reference['target_roots'],row['next_state'])[selected].mean())
                    else:bootstrap=float(values.max())
                    target+=agent.config.gamma*bootstrap
                a=(old[row['action']]-target)**2;b=(new[row['action']]-target)**2
                old_loss.append(a);new_loss.append(b);diffs.setdefault(row['block'],[]).append(b-a)
            for actual,expected in ((np.mean(old_loss),e['current_mse']),(np.mean(new_loss),e['candidate_mse']),
                                    (np.mean([np.mean(v) for v in diffs.values()]),e['loss_gate']['mean_difference'])):
                if not np.isclose(actual,expected,atol=1e-12,rtol=1e-10):raise ValueError('Bellman loss not reproduced')
            tv=float(np.mean([np.mean(v) for v in tvs.values()]))
            if not np.isclose(tv,e['policy_guard']['mean_tv'],atol=1e-12):raise ValueError('Policy TV not reproduced')
            old_node=next(n for p,n in walk(before[e['action']]) if p==e['path'])
            new_node=next(n for p,n in walk(after[e['action']]) if p==e['path'])
            if cell['mode']=='refit_only':
                old_nodes,new_nodes=list(walk(old_node)),list(walk(new_node))
                if len(old_nodes)!=len(new_nodes):raise ValueError('Refit changed the node count')
                for (p,a),(s,b) in zip(old_nodes,new_nodes):
                    if (p,a.literal,a.revision_uid,a.refinements_stats)!=(s,b.literal,b.revision_uid,b.refinements_stats):
                        raise ValueError('Refit modified structure or statistics')
            else:
                if not all(e[key]['accepted'] for key in ('loss_gate','policy_guard','return_guard')):
                    raise ValueError('An accepted repair bypassed a guard')
                if tv>agent.improvement.max_policy_tv+1e-12:raise ValueError('Policy trust limit exceeded')
                training=[row for row in agent.buffer if actions[row['action']]==e['action'] and any(
                    n.revision_uid==e['node'] for _,n in route(before[e['action']],row['state']))]
                reached={id(n):[] for _,n in walk(new_node)}
                for row in training:
                    for _,node in route(new_node,row['state']):reached[id(node)].append(old_node.predict(row['state']))
                for _,node in walk(new_node):
                    values=reached[id(node)]
                    if values and not min(values)-1e-12<=node.q_value<=max(values)+1e-12:
                        raise ValueError('Accepted repair left old prediction hull')
            records.append({'criterion':cell['criterion'],'seed':cell['seed'],'mode':cell['mode'],
                            'point':int(path.stem),'kind':e['kind'],'loss_reproduced':True,'policy_tv':tv})
    expected=sum(c['metrics'][k] for c in report['continuations'] for k in ('prunes_total','replacements_total','refits_total'))
    if len(records)!=expected:raise ValueError('Not all accepted parameter/structure changes audited')
    checked_write(output,{'complete':True,'campaign':digest(m),'accepted_changes':len(records),'records':records,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Reconstructed from independent probes and immediate pre/post weights; no extra RL training.'})
    print({'accepted_changes':len(records),'losses_and_policy_guards_reproduced':True},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.campaign,args.output)
