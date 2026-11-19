import csv,json,random,statistics,math,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=root/'analysis/baseline_anova_20261106';source=out/'anova_calendar_remote.json';raw=source.read_bytes();calendar=json.loads(raw)
audit_files={m:out/f'{m}_evaluations.csv' for m in ('historical','anova')}
use_audited_csv=all(p.exists() and p.stat().st_size>100 for p in audit_files.values())
hist=list(csv.DictReader((root/'analysis/baseline_10_seeds_20261102/evaluation_summary.csv').open()))
b={int(r['seed']):float(r['mean_return']) for r in hist};a={};entries=[]
for cell in calendar['cells']:
 seed=int(cell['cell'].split('/')[-1]);evaluations=[e for e in cell['evaluations_1m'] if e['group']=='final_development' and e['counter']==1_000_000];assert len(evaluations)==1
 e=evaluations[0];report=e['report'];eps=report['episodes'];assert len(eps)==10 and all(x['status']=='completed' for x in eps)
 values=[x['raw_return'] for x in eps];a[seed]=statistics.mean(values);assert a[seed]==report['mean_raw_return']
 entries.append({'seed':seed,'baseline_historical_f':b[seed],'baseline_anova':a[seed],'difference_anova_minus_historical':a[seed]-b[seed],'anova_evaluations':len(eps),'anova_unique_episode_returns':len(set(values)),'anova_unique_durations':len(set(x['steps'] for x in eps)),'anova_truncations':sum(x['truncated'] for x in eps),'anova_training_active_seconds':e['training_active_seconds']})
if use_audited_csv:
 audited={m:list(csv.DictReader(p.open())) for m,p in audit_files.items()}
 for method,expected in [('historical',b),('anova',a)]:
  for seed in range(710601,710611):
   rows=[r for r in audited[method] if int(r['seed'])==seed and r['group']=='final_development' and int(r['counter'])==1000000]
   assert len(rows)==10 and {int(r['eval_seed']) for r in rows}==set(range(712601,712611))
   assert all(r['status']=='completed' for r in rows)
   value=statistics.mean(float(r['raw_return']) for r in rows)
   assert value==expected[seed],(method,seed,value,expected[seed])
 # Values are explicitly rederived from audit outputs after asserting closure/old-summary agreement.
 b={seed:statistics.mean(float(r['raw_return']) for r in audited['historical'] if int(r['seed'])==seed and r['group']=='final_development' and int(r['counter'])==1000000) for seed in range(710601,710611)}
 a={seed:statistics.mean(float(r['raw_return']) for r in audited['anova'] if int(r['seed'])==seed and r['group']=='final_development' and int(r['counter'])==1000000) for seed in range(710601,710611)}
assert sorted(a)==sorted(b)==list(range(710601,710611))
seeds=sorted(a);av=[a[s] for s in seeds];bv=[b[s] for s in seeds];diff=[x-y for x,y in zip(av,bv)];n=len(seeds)
def quantile(v,p):
 index=(len(v)-1)*p;lo=math.floor(index);hi=math.ceil(index);return v[lo]+(v[hi]-v[lo])*(index-lo)
rng=random.Random(20261106);iters=100000;boot={'mean_difference':[],'median_paired_difference':[],'mean_ratio_anova_over_historical':[],'difference_of_medians':[],'anova_mean':[],'historical_mean':[]}
for _ in range(iters):
 ix=[rng.randrange(n) for _ in range(n)];aa=[av[i] for i in ix];bb=[bv[i] for i in ix];dd=[diff[i] for i in ix];am=statistics.mean(aa);bm=statistics.mean(bb)
 boot['mean_difference'].append(am-bm);boot['median_paired_difference'].append(statistics.median(dd));boot['difference_of_medians'].append(statistics.median(aa)-statistics.median(bb));boot['mean_ratio_anova_over_historical'].append(am/bm if bm else None);boot['anova_mean'].append(am);boot['historical_mean'].append(bm)
ci={}
for k,v in boot.items():
 finite=sorted(x for x in v if x is not None and math.isfinite(x));ci[k]={'low':quantile(finite,.025),'high':quantile(finite,.975),'finite_resamples':len(finite)}
wins=sum(d>0 for d in diff);losses=sum(d<0 for d in diff);ties=sum(d==0 for d in diff);m=wins+losses
upper=sum(math.comb(m,k) for k in range(wins,m+1))/2**m
sign_two=min(1,2*min(sum(math.comb(m,k) for k in range(wins,m+1))/2**m,sum(math.comb(m,k) for k in range(wins+1))/2**m))
ordered=sorted(zip(seeds,diff),key=lambda x:x[1]);leave_one=[{'excluded_seed':s,'remaining_n':n-1,'mean_difference':statistics.mean(d for t,d in zip(seeds,diff) if t!=s)} for s in seeds]
best_positive=[s for s,d in ordered[-2:]];remaining=[i for i,s in enumerate(seeds) if s not in best_positive]
history=list(csv.DictReader((root/'analysis/baseline_10_seeds_20261102/evaluation_history.csv').open()));history_stats={}
for k in (0,100000,500000,1000000):
 vals=[float(r['mean_return']) for r in history if r['group']=='intermediate' and int(r['interaction'])==k];history_stats[str(k)]={'mean':statistics.mean(vals),'median':statistics.median(vals),'training_seeds':len(vals)}
result={'scope':('Descriptive matched-seed comparison of final_development returns at 1M rederived from audit.py CSV outputs. audit.py verified individual evaluation/checkpoint units and hashes. This script asserts agreement with ANOVA closure and prior historical summary. Source versions differ; supplied protocol/source comparison is outside this script.' if use_audited_csv else 'Provisional descriptive matched-seed comparison from closure and prior summary; individual audit CSV unavailable.'),'sources':{'anova_closure':str(source),'anova_closure_sha256':hashlib.sha256(raw).hexdigest(),'historical_summary':str(root/'analysis/baseline_10_seeds_20261102/evaluation_summary.csv')},'experimental_unit':'trained agent / training seed, n=10 per criterion; not the 100 evaluation episodes','seeds':seeds,'paired_by_seed':entries,'summary':{'historical_mean':statistics.mean(bv),'anova_mean':statistics.mean(av),'historical_median':statistics.median(bv),'anova_median':statistics.median(av),'historical_sample_sd':statistics.stdev(bv),'anova_sample_sd':statistics.stdev(av),'mean_difference':statistics.mean(diff),'median_paired_difference':statistics.median(diff),'difference_of_medians':statistics.median(av)-statistics.median(bv),'mean_ratio_anova_over_historical':statistics.mean(av)/statistics.mean(bv),'wins_anova':wins,'losses_anova':losses,'ties':ties,'historical_return_at_most_3':sum(x<=3 for x in bv),'anova_return_at_most_3':sum(x<=3 for x in av),'historical_at_least_10':sum(x>=10 for x in bv),'anova_at_least_10':sum(x>=10 for x in av)},'paired_seed_percentile_bootstrap':{'iterations':iters,'random_seed':20261106,'quantile':'linear interpolation at (N-1)*p','confidence_level':.95,'resample':'same sampled seed indices applied jointly to both criteria; ten seeds sampled with replacement','intervals':ci,'caution':'Small n; percentile bootstrap and observed matching do not remove source-version confounding or provide population certainty.'},'exact_sign_test':{'wins':wins,'losses':losses,'ties_excluded':ties,'p_one_sided_anova_better':upper,'p_two_sided':sign_two,'null':'conditional on non-tie paired differences, positive and negative signs independent and equally likely','interpretation':'Descriptive exploratory test; no pre-registered multiplicity control or strict randomized attribution of criterion alone.'},'sensitivity':{'leave_one_seed_out':leave_one,'leave_one_out_mean_difference_range':[min(x['mean_difference'] for x in leave_one),max(x['mean_difference'] for x in leave_one)],'exclude_two_largest_positive_differences':{'excluded_seeds':best_positive,'remaining_n':len(remaining),'historical_mean':statistics.mean(bv[i] for i in remaining),'anova_mean':statistics.mean(av[i] for i in remaining),'mean_difference':statistics.mean(diff[i] for i in remaining),'note':'Post-hoc sensitivity, not an alternate unbiased effect estimate.'},'paired_difference_trimmed_mean_10pct':statistics.mean(sorted(diff)[1:-1]),'paired_difference_trimmed_mean_20pct':statistics.mean(sorted(diff)[2:-2]),'two_largest_positive_differences_sum':sum(d for s,d in ordered[-2:]),'total_paired_difference_sum':sum(diff)},'learning_curves':{'historical_intermediate':history_stats,'anova_intermediate_available_in_closure':False,'note':'Closure contains evaluations_1m only; do not infer ANOVA learning curve or regression across training from final snapshots.'},'limitations':['All ten ANOVA episodes per agent have identical return and duration; they do not supply ten independent trained-agent replicates.','Same seed IDs allow a descriptive matched comparison, but trajectories and episode boundaries diverge after different decisions.','Final_development is inspected development evaluation, not reserved test.','Statistical intervals ignore uncertainty due to protocol/source differences and deterministic-start coverage.']}
if use_audited_csv:
 result['sources']['audited_csv']={m:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for m,p in audit_files.items()}
 result['sources']['audit_script']=str(out/'audit.py')
 curves={}
 for method,rows in audited.items():
  curves[method]={}
  for checkpoint in (0,100000,500000,1000000):
   vals=[]
   for seed in range(710601,710611):
    rs=[r for r in rows if int(r['seed'])==seed and r['group']=='intermediate' and int(r['counter'])==checkpoint]
    assert len(rs)==2
    vals.append(statistics.mean(float(r['raw_return']) for r in rs))
   curves[method][str(checkpoint)]={'mean':statistics.mean(vals),'median':statistics.median(vals),'training_seeds':10}
 result['learning_curves']={'audited_intermediate':curves,'anova_intermediate_available_in_closure':False,'note':'Intermediate curves are rederived from audited individual evaluation CSV, not from the closure report. Sparse checkpoint observations do not establish a continuous trajectory.'}
(out/'comparison_stats.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'summary':result['summary'],'bootstrap':ci,'sign':result['exact_sign_test'],'sensitivity':result['sensitivity']},indent=2))
