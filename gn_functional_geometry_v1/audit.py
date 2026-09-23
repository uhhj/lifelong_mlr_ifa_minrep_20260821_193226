# Independent recomputation from raw CSV/NPZ, not production report helpers.
import csv,hashlib,json,numpy as np
from scipy.stats import spearmanr
from .common import ROOT,OUT,read,write,sha

def main():
 s=read(OUT/'DATA_SPLIT.json')['roles'];ctx=read(OUT/'EXPERIMENT_CONTEXT.json');checks={}
 for a,b in [('old_build','old_probe'),('old_gradient','old_probe'),('old_build','old_gradient'),('new_gradient','new_probe')]:
  checks[a+'_disjoint_'+b]=not ({r['trajectory_id'] for r in s[a]}&{r['trajectory_id'] for r in s[b]})
 checks['membership_and_sample_hashes']=all(sha(ROOT/r['origin'])==r['sha256'] for rows in s.values() for r in rows)
 plan=read(read(OUT/'DATA_SPLIT.json')['plan'])['immutable_feature_origins']
 checks['frozen_plan_membership']=all(plan[r['source_record_index']]==r['origin'] for rows in s.values() for r in rows)
 checks['formal_source_files_unchanged']=all(sha(r['path'])==r['sha256'] for r in ctx['source_files'])
 checks['selected_checkpoint_unchanged']=sha(ctx['checkpoint'])==ctx['checkpoint_sha256']
 checks['protocol_split_hash']=sha(OUT/'DATA_SPLIT.json')==read(OUT/'PROTOCOL.json')['split_sha256']
 a=np.load(OUT/'OLD_TASK_GN_BASIS.npz');lam=a['eigenvalues'];U=a['U'];num=read(OUT/'GN_NUMERICAL_CHECK.json')
 checks['basis_finite']=bool(np.isfinite(U).all() and np.isfinite(lam).all());checks['eigenvalues_sorted']=bool(np.all(np.diff(lam)<=0));checks['eigenvalues_nonnegative']=bool(np.min(lam)>=0)
 checks['trace_energy_not_truncated']=abs(float(lam.sum()/num['full_trace'])-read(OUT/'H2_REPORT.json')['captured_energy']['32'])<1e-12
 i=0;block=[]
 for name,size in zip(a['parameter_names'],a['parameter_sizes']):
  if 'logstd_layer' in name:block.append(float(np.max(abs(U[i:i+size]))))
  i+=size
 checks['variance_only_GN_rows_zero']=max(block)<1e-12
 h1=list(csv.DictReader((OUT/'H1_SENSITIVITY.csv').open()));h3=list(csv.DictReader((OUT/'H3_DIRECTION_PERTURBATION.csv').open()));raw={}
 for kind in ['GN','RANDOM']:
  rs=[r for r in h3 if r['subspace']==kind and float(r['scale'])==1. and r['label']!='WEAK'];pred=np.array([float(r['pred_old_change']) for r in rs]);true=np.array([float(r['measured_old_change']) for r in rs]);tol=read(OUT/'PROTOCOL.json')['H3']['loss_effect_tolerance']
  accuracy=float(np.mean((np.sign(pred)==np.sign(true))&(abs(true)>tol)));rho=float(spearmanr(pred,true)[0]);raw[kind]=dict(accuracy=accuracy,spearman=rho)
  reported=read(OUT/'H3_REPORT.json')['primary'][kind];checks[kind+'_raw_sign_recompute']=accuracy==reported['old_sign_accuracy'];checks[kind+'_raw_corr_recompute']=abs(rho-reported['spearman'])<1e-12
  # Same-evaluation-gradient Taylor discrepancy, entirely from existing observations.
  matches=np.array([float(r['matched_probe_pred_old']) for r in rs]);raw[kind]['median_absolute_taylor_error']=float(np.median(abs(true-matches)))
  raw[kind]['median_measured_abs_effect']=float(np.median(abs(true)))
 checks['equal_H1_norm']=all(abs(float(r['perturbation_norm'])-float(r['epsilon']))<1e-8 for r in h1)
 checks['equal_H3_norm']=all(abs(float(r['perturbation_norm'])-read(OUT/'PROTOCOL.json')['epsilon']*float(r['scale']))<1e-8 for r in h3 if r['label']!='WEAK')
 checks['zero_optimizer_steps']=read(OUT/'EFFICIENCY.json')['optimizer_steps_executed']==0
 checks['all_json_valid']=all(json.loads(p.read_text()) is not None for p in OUT.glob('*.json'))
 z=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,raw_recomputations=raw,training_active=False,formal_training_resumed=False)
 write(OUT/'INDEPENDENT_AUDIT.json',z);print(json.dumps(z))
 assert all(checks.values())
if __name__=='__main__':main()
