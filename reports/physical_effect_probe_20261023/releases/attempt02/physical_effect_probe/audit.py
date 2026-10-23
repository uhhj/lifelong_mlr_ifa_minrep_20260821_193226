"""Gate computed from complete held-out scalar evidence, no model or environment."""
import argparse
import numpy as np
from .common import *

def bootstrap_difference(base,candidate):
 keys=sorted(set(base)&set(candidate));assert len(keys)==10
 diff=np.array([base[k]['median_mm']-candidate[k]['median_mm'] for k in keys]);r=np.random.RandomState(410023)
 boot=diff[r.randint(0,len(diff),(10000,len(diff)))].mean(1)
 return dict(n_demos=len(keys),mean_improvement_mm=float(diff.mean()),ci95_mm=np.quantile(boot,[.025,.975]).tolist())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=41);ap.parse_args();verify()
 raw=js(OUT/'STATE_PROBE_RESULTS.json');assert len(raw['results'])==30
 native={(r['task'],r['model']):r['interventions'] for r in raw['results'] if r['train_view']==r['test_view']=='native'}
 report={}
 for kind in ('P2','P3'):
  checks=[]
  for task in (6,7):
   for field in ('object_eef','object_goal'):
    normal=native[task,kind]['normal'][field];base=native[task,'P0']['normal'][field]
    ci=bootstrap_difference(base['all']['by_demo'],normal['all']['by_demo'])
    dyn=normal['dynamic']['by_demo'];alt=native[task,kind]['latest_visual_previous'][field]['dynamic']['by_demo']
    lag_delta=float(np.mean([x['previous_target_median_mm']-x['median_mm'] for x in dyn.values()])) if dyn else None
    vision_delta=float(np.mean([alt[k]['median_mm']-v['median_mm'] for k,v in dyn.items()])) if dyn else None
    checks.append(dict(task=task,target=field,bootstrap=ci,median_mm=normal['all']['median_mm'],p90_mm=normal['all']['p90_mm'],dynamic_demos=len(dyn),current_over_lag_mm=lag_delta,latest_visual_benefit_mm=vision_delta,
     passed=bool(ci['ci95_mm'][0]>0 and normal['all']['median_mm']<=10 and normal['all']['p90_mm']<=25 and dyn and lag_delta>0 and vision_delta>0)))
  report[kind]=dict(passed=all(x['passed'] for x in checks),checks=checks)
 selected=next((k for k in ('P2','P3') if report[k]['passed']),None)
 d=dict(status='PASS_A' if selected else 'STOP_STATE_GATE',selected=selected,candidates=report,source_results_sha256=sha(OUT/'STATE_PROBE_RESULTS.json'),contract_sha256=sha(ROOT/'CONTRACT.json'),policy_updates=0,environment_steps=0)
 write(OUT/'A_GATE.json',d);log('A_gate',status=d['status'],selected=selected)
 print(json.dumps(d,indent=2))
if __name__=='__main__':main()
