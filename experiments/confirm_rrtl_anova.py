"""Prospective confirmation on fresh seeds; original seven-mode schema stays strict."""
import argparse
import json
import os
import time
from pathlib import Path

from bayesian_rrtl.online.protocol import runtime, source_hashes
from experiments import diagnose_rrtl_improvements as improvement
from experiments import diagnose_rrtl_revision as common
from experiments.compare_pong_progressive import StopRequest
from experiments.pong_analysis_runtime import checked_read, digest

MODES = ('baseline','slow_td','refit_only')
SCHEMA = 'rrtl_anova_confirmation_v1'


def validate_config(config):
    improvement.validate_config(config,expected_modes=MODES)
    if config['seeds'] != list(range(721101,721106)) or set(config['seed_roles'].values()) != {'confirmation'}:
        raise ValueError('Five fresh confirmation seeds required')
    if (config['parent_counter'] != 500000 or config['continuation_decisions'] != 500000
            or config['evaluation_interval'] != 25000 or config['save_interval'] != 25000
            or config['starts'] != 12 or config['evaluation_horizon'] != 3000):
        raise ValueError('Changed prospective design requires a new schema')


def load_campaign(output):
    output=Path(output);manifest=checked_read(output/'manifest.json')
    if (manifest['schema'] != SCHEMA or manifest['sources'] != source_hashes()
            or manifest['runtime'] != runtime('corrected_common')):
        raise ValueError('Confirmation source/runtime/schema mismatch')
    validate_config(manifest['config'])
    expected=[{'criterion':'anova','seed':s,'role':'confirmation'} for s in manifest['config']['seeds']]
    if manifest['groups'] != expected:
        raise ValueError('Confirmation group mismatch')
    return manifest,common.blob_read(output,manifest['start_bank'])


def prepare(config,parent_campaign,output):
    validate_config(json.loads(Path(config).read_text()))
    return common.prepare(config,parent_campaign,parent_campaign,output,
        expected_modes=MODES,manifest_schema=SCHEMA,existing_loader=load_campaign,
        extend_manifest=improvement.extend_manifest,criteria=('anova',))


def continue_group(output,index,stop):
    manifest,bank=load_campaign(output)
    if type(index) is not int or not 0 <= index < len(manifest['groups']):
        raise ValueError('Invalid confirmation group')
    group=manifest['groups'][index]
    parent=next(p for p in manifest['parents'] if p['seed']==group['seed'] and p['criterion']=='anova')
    for name in MODES:
        if not improvement.continue_one(output,manifest,bank,parent,name,stop):return False
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['parents','prepare','continue','report','analyze','audit'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path);parser.add_argument('--parents',type=Path)
    parser.add_argument('--group',type=int,default=0)
    args=parser.parse_args()
    with StopRequest() as stop:
        if args.stage=='parents':
            from bayesian_rrtl.online.protocol import freeze,authorize
            from experiments.compare_pong_progressive import train_cell
            raw=json.loads(args.config.read_text());freeze(raw,args.parents)
            costs=json.loads((args.output.parent/'cost_measurements.json').read_text())
            authorize(args.parents,1000000,'Prospective fresh-seed ANOVA comparison; measured Linux costs.',costs,game='Breakout')
            if not 0<=args.group<5:raise ValueError('Invalid parent group')
            complete=False
            while not stop.requested:
                remaining=int(os.environ['SLURM_JOB_END_TIME'])-time.time()-300
                if remaining<180:break
                status=train_cell(args.parents,str(args.group),attempt_seconds=min(1740,remaining),stop=stop)
                if status['state']=='completed' and status['counter']==1000000:
                    complete=True;break
                if status.get('reason') not in ('attempt_time_exhausted','interrupted_signal','administrative_pause'):
                    raise RuntimeError('Parent stopped; review before retry: '+str(status.get('reason')))
        elif args.stage=='prepare':prepare(args.config,args.parents,args.output);complete=True
        elif args.stage=='continue':complete=continue_group(args.output,args.group,stop)
        elif args.stage=='report':complete=improvement.report(args.output,loader=load_campaign)
        elif args.stage=='audit':
            import importlib.util
            path=Path(__file__).resolve().parents[1]/'analysis/confirmation_20261109/audit_guards.py'
            spec=importlib.util.spec_from_file_location('confirmation_guard_audit',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.audit(args.output,args.output.parent/'confirmation_guard_audit.json',loader=load_campaign)
            complete=True
        else:
            from experiments.analyze_rrtl_improvements import analyze
            analyze(args.output,args.output.parent/'analysis',loader=load_campaign);complete=True
    raise SystemExit(0 if complete else 3)


if __name__=='__main__':main()
