"""Reproduce two first-intervention prefixes and descriptive TD intervals."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
ROOT = REPO/'results_cluster/rrtl_improvements_20261108_01/campaign'
FIELDS = ('interaction','transition_id','episode_id','step_id','episode_seed','epsilon','action',
          'reward','raw_reward','terminated','truncated','episode_end','raw_episode_return','td')


def rows(folder, limit):
    status = json.loads((folder/'status.json').read_text())['payload']
    n = 0
    for ref in status['traces']:
        raw = (folder/ref['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref['sha256']
        for line in gzip.decompress(raw).splitlines():
            if n == limit:
                return
            n += 1
            yield json.loads(line)
    assert n >= limit


records = []
for criterion, seed, method, control in [('historical_f',710607,'guarded','baseline'),
                                         ('anova',710602,'stable_guarded','stable_td')]:
    a = ROOT/'continuations'/criterion/str(seed)/method
    b = a.parent/control
    n = 0
    for x, y in zip(rows(a,25000), rows(b,25000)):
        assert all(x[k] == y[k] for k in FIELDS)
        assert all(x['diagnostic'][k] == y['diagnostic'][k] for k in ['q_before','q_after'])
        n += 1
    assert n == 25000
    records.append({'criterion':criterion,'seed':seed,'method':method,'control':control,
        'exact_prefix_transitions':n,
        'scope':'Actions, TD, Q before/after, rewards and episode fields coincide until the first accepted structural intervention at 525k. Later returns include subsequent learning.'})
(OUT/'first_intervention_prefixes.json').write_text(json.dumps(records,indent=2)+'\n')

matched = list(csv.DictReader((OUT/'matched_vs_baseline.csv').open()))
intervals = []
for criterion in ['historical_f','anova']:
    values = [float(r['mean_difference']) for r in matched if r['criterion']==criterion and r['mode']=='slow_td']
    assert len(values) == 5
    rng = random.Random(20261109)
    samples = sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(50000))
    intervals.append({'criterion':criterion,'variant':'slow_td','seeds':5,
        'mean_paired_difference':statistics.mean(values),'bootstrap_descriptive_95':[samples[1250],samples[48749]],
        'scope':'Post-hoc descriptive interval by training seed; five selected seeds, no confirmatory significance claim.'})
(OUT/'slow_td_uncertainty.json').write_text(json.dumps(intervals,indent=2)+'\n')
print(json.dumps({'prefix_checks':records,'descriptive_intervals':intervals},indent=2))
