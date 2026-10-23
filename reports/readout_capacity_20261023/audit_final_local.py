"""Local independent errors/quantiles/demo cluster CI/gate from scalar-only evidence."""
import sys,json,hashlib,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;S=Path(sys.argv[1]);E=Path(sys.argv[2]);I=E;meta=S/'metadata'
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=js(E/'EVIDENCE.json');independent=js(I/'EVIDENCE.json');raw=js(meta/(manifest['phase']+'_RESULTS.json'));gate=js(meta/'A_GATE.json');snap=js(S/'EVIDENCE.json')
assert sha(meta/(manifest['phase']+'_RESULTS.json'))==manifest['source_results_sha256']==independent['source_results_sha256']
assert snap['controller_exit']=='0' and not snap['gpu'].strip() and not snap['hash_bad']
assert all(x=='0' for x in snap['phase_exits'].values())
assert manifest['status']=='PASS_COMPLETE_CP_SELECTION_PARTITIONS_SCALAR_EXPORT'
assert manifest['contract_sha256']==gate['contract_sha256']
for name,md in js(S/'DOWNLOAD.json').items():assert sha(meta/name)==md['sha256']
cases={};count=0
for name,md in manifest['case_files'].items():
 path=E/(name+'.json');assert sha(path)==md['sha256'] and path.stat().st_size==md['bytes'];x=js(path)
 ref={'interventions':raw['summaries']['%s_b%d_task%d'%(x['model'],x['budget'],x['task'])]}
 demos=np.array(x['demo_ids']);dyn=np.array(x['dynamic'],dtype=bool);assert len(x['ids'])==len(set(x['ids']))
 for mode,arrays in x['modes'].items():
  err=np.array(arrays['error_mm']);prev=np.array(arrays['previous_target_error_mm']);assert err.shape==(len(demos),3) and prev.shape==(len(demos),2) and np.isfinite(err).all()
  for j,target in enumerate(('object_eef','object_goal','aperture')):
   for s,mask in [('all',np.ones(len(demos),bool)),('dynamic',dyn),('static',~dyn)]:
    expected=ref['interventions'][mode][target][s];assert expected['n']==int(mask.sum())
    assert math.isclose(expected['median_mm'],float(np.median(err[mask,j])),rel_tol=1e-5,abs_tol=1e-4)
    assert math.isclose(expected['p90_mm'],float(np.quantile(err[mask,j],.9)),rel_tol=1e-5,abs_tol=1e-4)
    for d,v in expected['by_demo'].items():
     maskd=mask&(demos==d);assert int(maskd.sum())==v['n'];assert math.isclose(v['median_mm'],float(np.median(err[maskd,j])),rel_tol=1e-5,abs_tol=1e-4)
     if j<2:assert math.isclose(v['previous_target_median_mm'],float(np.median(prev[maskd,j])),rel_tol=1e-5,abs_tol=1e-4)
  count+=len(demos)
 cases[x['task'],x['model'],x['budget']]=x
assert manifest['phase']=='CAPACITY' and len(cases)==18 and len(manifest['checkpoint_sha256'])==90
assert manifest['actual_cost']['fits']==30 and manifest['actual_cost']['epochs']==6000 and manifest['actual_cost']['updates']==49800
checks={}
for kind in ('G','S'):
 rows=[]
 for task in (6,7):
  x=cases[task,kind,200];b=cases[task,'P0',200];ids=np.array(x['demo_ids']);bd=np.array(b['demo_ids']);dyn=np.array(x['dynamic'],bool)
  e=np.array(x['modes']['normal']['error_mm']);pe=np.array(x['modes']['normal']['previous_target_error_mm']);z=np.array(x['modes']['latest_visual_previous']['error_mm']);be=np.array(b['modes']['normal']['error_mm'])
  for j,target in enumerate(('object_eef','object_goal')):
   demos=sorted(set(ids));delta=np.array([np.median(be[bd==d,j])-np.median(e[ids==d,j]) for d in demos]);rng=np.random.RandomState(410023);ci=np.quantile(delta[rng.randint(0,10,(10000,10))].mean(1),[.025,.975])
   dd=sorted(set(ids[dyn]));lag=np.mean([np.median(pe[dyn&(ids==d),j])-np.median(e[dyn&(ids==d),j]) for d in dd]);vis=np.mean([np.median(z[dyn&(ids==d),j])-np.median(e[dyn&(ids==d),j]) for d in dd])
   median=float(np.median(e[:,j]));p90=float(np.quantile(e[:,j],.9));passed=bool(ci[0]>0 and median<=10 and p90<=25 and len(dd)>0 and lag>0 and vis>0)
   rows.append(dict(task=task,target=target,median_mm=median,p90_mm=p90,ci95_improvement_mm=ci.tolist(),dynamic_lag_advantage_mm=float(lag),dynamic_visual_advantage_mm=float(vis),passed=passed))
 checks[kind]=dict(passed=all(v['passed'] for v in rows),checks=rows)
 assert checks[kind]['passed']==gate['candidates'][kind]['passed']
 for computed,recorded in zip(rows,gate['candidates'][kind]['checks']):
  assert computed['task']==recorded['task'] and computed['target']==recorded['target'] and computed['passed']==recorded['passed']
  for field in ('median_mm','p90_mm','dynamic_lag_advantage_mm','dynamic_visual_advantage_mm'):assert math.isclose(computed[field],recorded[field],abs_tol=1e-4,rel_tol=1e-5)
  assert np.allclose(computed['ci95_improvement_mm'],recorded['bootstrap']['ci95_mm'],atol=1e-4,rtol=1e-5)
selected=next((k for k in ('G','S') if checks[k]['passed']),None);assert selected==gate['selected']==independent['selected']
result=dict(status='PASS_LOCAL_COMPLETE_A',selected=selected,scientific_decision=gate['status'],checks=checks,complete_cases=len(manifest['case_files']),scalar_error_rows_recomputed=count,source_results_sha256=sha(meta/(manifest['phase']+'_RESULTS.json')),scalar_manifest_sha256=sha(E/'EVIDENCE.json'),server_independent_sha256=sha(I/'EVIDENCE.json'),snapshot_sha256=sha(S/'EVIDENCE.json'),actual_cost=independent['actual_cost'],policy_updates=0,formal_episodes=0)
out=R/('final_A_'+manifest['phase']+'_20261023');out.mkdir(exist_ok=True);(out/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
