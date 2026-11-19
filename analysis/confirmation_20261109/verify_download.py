"""Verify all downloaded prospective-confirmation results and trace summaries.

Run with the frozen scientific source on PYTHONPATH. This verifier does not
deserialize checkpoints or replace the independent cluster-side refit audit.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from experiments.improvement_trace_summary import trace_summary


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checked(path):
    raw = json.loads(path.read_text())
    assert raw['schema'] == 'pong_analysis_unit_v2' and raw['sha256'] == digest(raw['payload']), str(path)
    return raw['payload']


def verify(campaign, analysis):
    manifest = checked(campaign/'manifest.json')
    assert manifest['schema'] == 'rrtl_anova_confirmation_v1'
    report = checked(campaign/'report.json')
    assert report['complete'] and not report['missing'] and report['campaign'] == digest(manifest)
    config = manifest['config']
    assert config['seeds'] == list(range(721101,721106))
    assert config['modes'] == ['baseline','slow_td','refit_only']
    assert config['parent_counter'] == 500000 and config['continuation_decisions'] == 500000
    assert config['save_interval'] == config['evaluation_interval'] == 25000
    assert config['evaluation_horizon'] == 3000 and config['starts'] == 12
    assert manifest['groups'] == [
        {'criterion':'anova','seed':seed,'role':'confirmation'} for seed in config['seeds']]
    points = list(range(0,500001,25000))
    starts = {v['id']:v for v in checked(campaign/'starts.json')}
    banks = [list(starts.values()), checked(campaign/'validation_starts.json')]
    banks.extend(checked(campaign/'return_starts'/f'{point}.json') for point in points[1:])
    seen = set()
    for bank in banks:
        ram = {v['ram_sha256'] for v in bank}
        assert len(ram) == len(bank) and not seen & ram
        seen.update(ram)
    assert len(starts) == 12
    cells = {(v['criterion'],v['seed'],v['mode']):v for v in report['continuations']}
    expected = {(g['criterion'],g['seed'],mode) for g in manifest['groups'] for mode in config['modes']}
    assert set(cells) == expected and len(cells) == 15
    saved_windows = list(csv.DictReader((analysis/'training_windows.csv').open()))
    window_map = {(v['criterion'],int(v['seed']),v['mode'],int(v['point_start'])):v for v in saved_windows}
    assert len(window_map) == len(saved_windows) == 1500
    trace_rows = evaluations = segments = reviews = windows = 0
    records = []
    for key, cell in cells.items():
        criterion, seed, mode = key
        parent = next(p for p in manifest['parents'] if (p['criterion'],p['seed']) == (criterion,seed))
        folder = campaign/'continuations'/criterion/str(seed)/mode
        status = checked(folder/'status.json')
        identity = {'campaign':digest(manifest), 'parent':parent['checkpoint'], 'criterion':criterion,
                    'seed':seed, 'mode':mode, 'parent_counter':parent['counter']}
        assert status['identity'] == identity and status['state'] == 'completed'
        assert status['counter'] == cell['counter'] == 1000000 and status['continuation_decisions'] == 500000
        assert status['metrics'] == cell['metrics']
        assert set(status['checkpoints']) == {str(p) for p in points}
        recorded = {(v['point'],v['start']):v for v in cell['evaluations']}
        assert len(recorded) == len(cell['evaluations']) == 252
        for point in points:
            for start_id, start in starts.items():
                path = folder/'evaluations'/str(point)/str(start_id)/'state.json'
                saved, ei = checked(path), checked(path.with_name('identity.json'))
                want = {**identity, 'checkpoint':status['checkpoints'][str(point)], 'point':point,
                        'start_bank':manifest['start_bank'], 'start':start_id, 'epsilon':0.,
                        'horizon':3000, 'evaluation_seed':config['evaluation_seed']+start_id}
                assert ei == want and saved['identity_sha256'] == digest(ei) and saved['status'] == 'completed'
                result = saved['result']
                assert {'point':point, **result} == recorded[point,start_id]
                assert result['physical_hash'] == start['physical_hash'] and result['start'] == start_id
                assert result['horizon'] == 3000 and result['seed'] == want['evaluation_seed'] and result['epsilon'] == 0
                assert 0 < result['steps'] <= 3000 and sum(result['actions'].values()) == result['steps']
                evaluations += 1
        events = []
        for path in sorted((folder/'audits').glob('*.json'), key=lambda p:int(p.stem)):
            if path.stem not in status['checkpoints']:
                continue
            audit = checked(path)
            assert audit['identity'] == identity and audit['counter'] == 500000+int(path.stem)
            events.extend(audit['events'])
        assert events == cell['events']
        for path in (folder/'reviews').glob('*.json'):
            if path.stem not in status['checkpoints']:
                continue
            saved = checked(path)
            assert saved['identity'] == identity and not saved['validation_used_for_td'] and not saved['evaluation_bank_used_for_gate']
            for event in saved['reviews']:
                if event['gate']['accepted'] and mode in ['guarded','stable_guarded']:
                    assert all(event[k]['accepted'] for k in ['loss_gate','policy_guard','return_guard'])
            reviews += 1
        reproduced = trace_summary(folder,status,500000)
        assert len(reproduced) == 100 and sum(v['decisions'] for v in reproduced) == 500000
        for row in reproduced:
            saved = window_map[key+(row['point_start'],)]
            for field, value in row.items():
                if value is None:
                    assert saved[field] == ''
                elif isinstance(value,float):
                    assert math.isclose(float(saved[field]),value,rel_tol=1e-12,abs_tol=1e-12), (key,field)
                else:
                    assert int(saved[field]) == value, (key,field)
        n = sum(v['decisions'] for v in reproduced)
        trace_rows += n; segments += len(status['traces']); windows += len(reproduced)
        records.append({'criterion':criterion,'seed':seed,'mode':mode,'trace_rows':n,
                        'evaluation_rollouts':252,'status_payload_sha256':digest(status)})
        print(json.dumps({'verified':[criterion,seed,mode],'trace_rows':trace_rows,'evaluations':evaluations}),flush=True)
    assert trace_rows == 7500000 and evaluations == 3780 and windows == 1500
    result = {'complete':True, 'campaign':digest(manifest), 'continuations':len(records),
              'verified_trace_rows':trace_rows,'verified_trace_segments':segments,
              'verified_evaluation_rollouts':evaluations,'reproduced_training_windows':windows,
              'verified_review_documents':reviews,'banks_ram_disjoint':True,'records':records,
              'scope':'All downloaded committed JSON, trace bytes/rows and training summaries. Large model/probe/emulator blobs remain on cluster; their report and independent refit audit must also pass the closure verifier.'}
    raw = {'schema':'rrtl_confirmation_local_download_verification_v1','payload':result,'sha256':digest(result)}
    (analysis/'local_data_verification.json').write_text(json.dumps(raw,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    a = p.parse_args();verify(a.campaign,a.analysis)
