"""C13/C14 diagnostic controls and exact detector/relational process continuation."""
import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import pickle

from bayesian_rrtl.online.environments import CommonGameFactory,GameEnvironmentConfig
from experiments.atari_fact_controls import FactController
from experiments.pong_analysis_runtime import (atomic_bytes,atomic_json,bytes_hash,episode)


def signature(env, action):
    facts,reward,raw,terminal,truncated=env.step(action)
    return {'objects':asdict(env.backend.current_frame),
            'detector':env.backend.detector_state(),
            'facts':sorted([list(f) for f in facts],key=repr),
            'reward':reward,'raw':raw,'terminated':terminal,'truncated':truncated,
            'lives':env.last_lives}


def child(config, checkpoint, actions, output):
    env=CommonGameFactory(GameEnvironmentConfig(**config))()
    try:
        env.restore(pickle.loads(Path(checkpoint).read_bytes()))
        rows=[]
        for a in actions:
            if env.done:
                break
            rows.append(signature(env,a))
        atomic_bytes(output,pickle.dumps(rows))
    finally:
        env.close()


def validate(config, output, *, seed=640101, prefix=32, future=64):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    factory=CommonGameFactory(config)
    env=factory()
    try:
        env.reset(seed=seed)
        for i in range(prefix):
            if env.done:
                raise ValueError('Prefix ended before planned snapshot')
            env.step(i%len(env.actions))
        saved=env.snapshot()
        checkpoint=output/'environment.pkl'
        atomic_bytes(checkpoint,pickle.dumps(saved))
        actions=[(i*3+1)%len(env.actions) for i in range(future)]
        reference=[]
        for a in actions:
            if env.done:
                break
            reference.append(signature(env,a))
        p=mp.get_context('spawn').Process(target=child,args=(asdict(config),checkpoint,actions,output/'child.pkl'))
        p.start()
        p.join(120)
        if p.is_alive():
            p.kill()
            p.join(10)
            raise TimeoutError('Bounded continuation worker timed out')
        if p.exitcode != 0:
            raise RuntimeError('Continuation worker failed')
        resumed=pickle.loads((output/'child.pkl').read_bytes())
        assert reference == resumed, 'Detector/facts/reward/flags differ after process restore'
        report={'game':config.game,'integration':'verified','exact_resume_between_processes':True,
                'compared_steps':len(reference),'config':asdict(config),'environment':saved['manifest'],
                'checkpoint_sha256':bytes_hash(checkpoint.read_bytes()),
                'trace_sha256':bytes_hash(pickle.dumps(reference)),
                'learning_usefulness':'not measured by continuity', 'test_reserved_executed':False}
        atomic_json(output/'continuity.json',report)
        return report
    finally:
        env.close()


def controls(config, output, *, seeds=(640201,640202), seconds=120):
    output=Path(output)
    factory=CommonGameFactory(config)
    env=factory()
    try:
        runtime_manifest=env.runtime_manifest()
    finally:
        env.close()
    rows=[]
    for policy in ('noop','random','heuristic'):
        for seed in seeds:
            identity={'schema':'atari_fact_controls_v1','environment':config.manifest(),
                      'runtime':runtime_manifest,'policy':policy,'seed':seed,
                      'policy_code':bytes_hash(Path(__file__).with_name('atari_fact_controls.py').read_bytes())}
            row=episode(factory,FactController(config.game,policy),output/policy/str(seed),identity,
                        seed=seed,horizon=config.horizon,seconds=seconds,rng_seed=seed,pong_points=False)
            rows.append({'policy':policy,**row})
    report={'game':config.game,'rows':rows,
            'state':'completed' if all(r['status']=='completed' for r in rows) else 'partial',
            'interpretation':'Fact-only diagnostic control; bbox contact is not certified collision; no competence guarantee.'}
    atomic_json(output/'controls.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True,help='GameEnvironmentConfig JSON')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('continuity','controls'),default='continuity')
    args=parser.parse_args()
    config=GameEnvironmentConfig(**json.loads(args.config.read_text()))
    result=validate(config,args.output) if args.stage=='continuity' else controls(config,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('environment','rows')},indent=2))


if __name__=='__main__':
    main()
