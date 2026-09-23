import csv,json,numpy as np
from .common import *
def main():
 p=read(OUT/'PROTOCOL.json');checks={}
 checks['all_parent_bases_and_draws_unchanged']=all(sha(a['path'])==a['sha256'] for a in p['assets'].values())
 c=read(OUT/'EXPERIMENT_CONTEXT.json');checks['checkpoint_unchanged']=sha(c['checkpoint'])==c['checkpoint_sha256'];checks['formal_sources_unchanged']=all(sha(r['path'])==r['sha256'] for r in c['source_files'])
 checks['split_unchanged']=sha(OUT/'DATA_SPLIT.json')==p['split_sha256']==sha(OLD/'DATA_SPLIT.json')
 a=list(csv.DictReader((OUT/'A_DIRECTIONAL_CLOSURE.csv').open()));groups={}
 for row in a:
  calc=abs(float(row['finite_difference'])-float(row['analytic']))/max(abs(float(row['analytic'])),1e-8*float(row['gradient_norm']))
  assert abs(calc-float(row['relative_error']))<=1e-12*max(1,calc)
  groups.setdefault((row['direction_type'],row['direction'],row['task']),[]).append(row)
 passed=0
 for rows in groups.values():
  assert len(rows)==8
  success=any(float(x['h_factor'])<=.001 and float(x['relative_error'])<=1e-4 and float(y['relative_error'])<=1e-4 for x,y in zip(rows,rows[1:]));passed+=success
 checks['A_70_groups_independently_closed']=passed==70==read(OUT/'A_REPORT.json')['passed_groups']
 b=list(csv.DictReader((OUT/'B_SUBSPACE_EFFECTS.csv').open()));windows=list(csv.DictReader((OUT/'B_PER_WINDOW_EFFECTS.csv').open()));grouped={}
 for row in windows:grouped.setdefault((row['geometry'],row['rank'],row['scale'],row['role']),[]).append(float(row['delta_nll']))
 errors=[]
 for row in b:
  for role,count in [('old_probe',128),('new_probe',64)]:
   v=grouped[(row['geometry'],row['rank'],row['scale'],role)];assert len(v)==count
   errors.append(abs(float(np.mean(v))-float(row['measured_'+role.split('_')[0]])))
  eps=p['epsilon0']*float(row['scale']);pred=-eps*float(row['score_old_relation'])/float(row['projected_new_gradient_norm'])
  assert abs(pred-float(row['pred_old_cross']))<1e-10*max(1,abs(pred))
 checks['B_all_24_equal_norm_cases']=len(b)==24 and all(abs(float(r['perturbation_norm'])-p['epsilon0']*float(r['scale']))<1e-10 for r in b)
 checks['B_prediction_scaling_identity']=True;checks['B_per_window_recomputed']=max(errors)<1e-10
 rotations=read(OUT/'BASIS_INVARIANCE.json')['rows'];bounds=[2*r['relative_projected_vector_difference']/(1-r['relative_projected_vector_difference']) for r in rotations]
 checks['B_rotation_and_normalized_direction_bound']=all(r['passed'] for r in rotations) and max(bounds)<1e-10
 checks['Fisher_extension']=read(OUT/'FISHER_EXTENSION_CHECK.json')['status']=='PASS'
 checks['zero_optimizer_steps']=read(OUT/'EFFICIENCY.json')['optimizer_steps_executed']==0
 checks['valid_json']=all(json.loads(path.read_text()) is not None for path in OUT.glob('*.json'))
 checks={k:bool(v) for k,v in checks.items()}
 write(OUT/'INDEPENDENT_AUDIT.json',dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,max_window_aggregation_difference=max(errors),max_rotation_normalized_direction_error_bound=max(bounds),training_active=False,formal_training_resumed=False));assert all(checks.values());print('INDEPENDENT_AUDIT_PASS')
if __name__=='__main__':main()
