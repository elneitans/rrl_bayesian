"""Consecutive-fit policy drift weighted by actual final replay visitation."""
from pathlib import Path
import sys,argparse,csv
from collections import Counter
import numpy as np
from analyze_checkpoints import load,checked_json,EncodedBatch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for folder in sorted((args.source/'Breakout/ocatari/bart').glob('*')):
        status=checked_json(folder/'status.json')
        x=load(folder/status['checkpoint']['path']);ag=x['session']['collector']['agent'];counts=Counter();states={}
        for tr in ag.replay.transitions:
            v=tr.state.values[0];k=v.tobytes();counts[k]+=1;states[k]=v
        keys=sorted(counts);v=np.array([states[k] for k in keys]);w=np.array([counts[k] for k in keys],float);w/=w.sum()
        batch=EncodedBatch(v,v!=-1,ag.encoder.catalog,ag.encoder.catalog_hash)
        q=ag.model.predict(batch);old=ag.target_model.predict(batch)
        a=q==q.max(1,keepdims=True);b=old==old.max(1,keepdims=True)
        flip=~(a&b).any(1)
        margin=np.sort(q,axis=1)[:,-1]-np.sort(q,axis=1)[:,-2]
        rows.append({'seed':int(folder.name),'counter':x['counter'],'from_model':ag.target_model.model_id,
            'to_model':ag.model.model_id,'replay_rows':len(ag.replay),'unique_states':len(batch),
            'replay_weighted_greedy_flip_pct':float(100*w@flip),
            'unweighted_unique_state_flip_pct':float(100*np.mean(flip)),
            'replay_weighted_q_rmse':float(np.sqrt(w@np.mean((q-old)**2,axis=1))),
            'replay_weighted_action_margin_mean':float(w@margin),
            'q_abs_max_on_all_replay_states':float(abs(q).max())})
        print(rows[-1],flush=True)
    with (args.output/'replay_weighted_consistency.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

if __name__=='__main__':main()
