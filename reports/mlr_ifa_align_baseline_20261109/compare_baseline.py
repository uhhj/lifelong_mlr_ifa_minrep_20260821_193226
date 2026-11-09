"""Compare all accepted new outcomes with all six bound Random baselines."""
from pathlib import Path
import json,sys,statistics,hashlib
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'))
a=js(D/'LOCAL_FINAL_AUDIT.json');assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
b=js(R/'BASELINE_BINDING.json');old=js(R/'BASELINE_OUTCOMES.json')['cases'];cases={};negative=[]
for case,r in a['cases'].items():
 baseline=b['cases'][case];delta={k:r['metrics'][k]['percent']-v['percent'] for k,v in baseline['baseline_metrics'].items()}
 index=lambda x:(x['stage'],x['task'],x['initial_state_index'])
 o={index(x):x for x in old[case]};n={index(x):x for x in r['records']};assert o.keys()==n.keys() and len(o)==200
 assert all(o[k]['initial_state_sha256']==n[k]['initial_state_sha256'] and o[k]['seed']==n[k]['seed'] for k in o)
 sf=sum(n[k]['success'] and not o[k]['success'] for k in o);fs=sum(o[k]['success'] and not n[k]['success'] for k in o)
 cells={k:v-baseline['baseline_matrix'][k] for k,v in r['matrix'].items()}
 negative += [dict(case=case,cell=k,delta_pp=v) for k,v in cells.items() if v<0]
 cases[case]=dict(new_metrics=r['metrics'],baseline_metrics=baseline['baseline_metrics'],delta_pp=delta,cell_delta_pp=cells,new_success_baseline_failure=sf,new_failure_baseline_success=fs,paired_initial_states=200,historical_gpu=baseline['historical_gpu'])
suites={}
for suite in ['OBJECT','GOAL']:
 group=[cases[suite+'_'+str(s)] for s in [41,42,43]];suites[suite]={}
 for name in ['new_metrics','baseline_metrics','delta_pp']:
  suites[suite][name]={}
  for metric in group[0][name]:
   values=[r[name][metric] if name=='delta_pp' else r[name][metric]['percent'] for r in group]
   suites[suite][name][metric]=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values))
out=dict(status='PASS_SIX_CASE_RANDOM_ALIGN_COMPARISON',source_final_sha256=hashlib.sha256((D/'LOCAL_FINAL_AUDIT.json').read_bytes()).hexdigest(),cases=cases,suites=suites,all_negative_cells=negative,limits=['Same original Random members and optimizer budget; ALIGN changes targets/masks only within the new adapted path.','Historical hardware/global action random streams are not a same-hardware paired causal experiment.','Development-set reuse and unresolved pretraining provenance remain.','All six runs and negative cells included; no parameter/seed/outcome selection.'])
(D/'BASELINE_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
lines=['# Random MLR+IFA + ALIGN versus accepted Random MLR+IFA','','All metrics in percent; NBT lower is better. No baseline reruns.','','|Suite / method|AUC|FWT|NBT|OldFinal|NewFinal|FinalAll|','|---|---:|---:|---:|---:|---:|---:|']
for suite,values in suites.items():
 for method in ['baseline_metrics','new_metrics','delta_pp']:
  lines.append('|'+suite+' / '+method+'|'+'|'.join('%.4f ± %.4f'%(values[method][m]['mean'],values[method][m]['sample_sd']) for m in ['AUC','FWT','NBT','OldFinal','NewFinal','FinalAll'])+'|')
lines+=['','## All lower-success cells','','|Case|Stage_task|Delta (pp)|','|---|---|---:|']+[f"|{r['case']}|{r['cell']}|{r['delta_pp']:.4f}|" for r in negative]
lines+=['','## Boundaries','']+out['limits']
(D/'BASELINE_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(out['status'])
