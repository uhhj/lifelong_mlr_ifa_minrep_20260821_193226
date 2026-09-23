import csv,json,numpy as np
from scipy.stats import spearmanr,pearsonr
from .common import *
def main():
 p=read(OUT/'PROTOCOL.json');checks={}
 checks['same_split']=sha(OUT/'DATA_SPLIT.json')==p['split_sha256']==sha(OLD/'DATA_SPLIT.json')
 checks['old_basis_unmodified']=sha(p['mean_basis_path'])==p['mean_basis_sha256']
 context=read(OUT/'EXPERIMENT_CONTEXT.json');checks['checkpoint_unmodified']=sha(context['checkpoint'])==p['checkpoint_sha256']
 checks['formal_model_sources_unchanged']=all(sha(r['path'])==r['sha256'] for r in context['source_files'])
 checks['parent_observations_reproduced']=read(OUT/'PARENT_PARITY.json')['status']=='PASS'
 checks['fisher_numerical']=read(OUT/'FISHER_NUMERICAL_CHECK.json')['status']=='PASS'
 a=np.load(OUT/'FISHER_MODEL_DRAWS.npz');checks['model_draws_shape_finite']=a['actions'].shape==(128,8,7) and bool(np.isfinite(a['actions']).all())
 data=list(csv.DictReader((OUT/'DIRECTION_EFFECTS.csv').open()));window=list(csv.DictReader((OUT/'PER_WINDOW_EFFECTS.csv').open()));metrics=list(csv.DictReader((OUT/'METRICS.csv').open()))
 checks['exact_two_by_four']=len(data)==2*16*4 and set(float(r['scale']) for r in data)==set(p['scales'])
 checks['equal_norms']=all(abs(float(r['norm'])-p['epsilon0']*float(r['scale']))<1e-8 for r in data if r['weak_new_projection']=='False')
 grouped={}
 for r in window:grouped.setdefault((r['geometry'],r['direction'],r['scale'],r['role']),[]).append(float(r['delta_nll']))
 errors=[]
 for r in data:
  for role,n in [('old_probe',128),('new_probe',64)]:
   vals=grouped[(r['geometry'],r['direction'],r['scale'],role)];assert len(vals)==n
   errors.append(abs(np.mean(vals)-float(r['measured_'+role.split('_')[0]])))
 checks['per_window_aggregation']=max(errors)<1e-10
 for r in metrics:
  subset=[z for z in data if z['geometry']==r['geometry'] and float(z['scale'])==float(r['scale'])]
  pred=np.array([float(z['pred_'+r['task']+'_'+r['predictor']]) for z in subset]);true=np.array([float(z['measured_'+r['task']]) for z in subset]);tol=p['loss_effect_tolerance']
  elig=(abs(pred)>tol)&(abs(true)>tol)&np.array([z['weak_new_projection']=='False' for z in subset]);correct=int(np.sum(elig&(np.sign(pred)==np.sign(true))))
  assert correct==int(r['correct']);assert abs(float(spearmanr(pred,true)[0])-float(r['spearman']))<1e-12;assert abs(float(pearsonr(pred,true)[0])-float(r['pearson']))<1e-12
 checks['all_metric_rows_independently_recomputed']=True
 checks['zero_optimizer_steps']=read(OUT/'EFFICIENCY.json')['optimizer_steps_executed']==0
 checks['all_json_parse']=all(json.loads(f.read_text()) is not None for f in OUT.glob('*.json'))
 checks={k:bool(v) for k,v in checks.items()}
 result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,max_window_aggregation_difference=max(errors),training_active=False,formal_training_resumed=False)
 write(OUT/'INDEPENDENT_AUDIT.json',result);print(json.dumps(result));assert all(checks.values())
if __name__=='__main__':main()
