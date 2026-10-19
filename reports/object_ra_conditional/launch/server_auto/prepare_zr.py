"""Create only the selected authorized branch, with immutable concrete parent binding."""
from pathlib import Path
import json,shutil,hashlib,sys,time,os
N=Path(__file__).resolve().parent
Q=Path('/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019')
def parent_spec(branch,seed):
 assert (branch,seed) in [(b,s) for b,seeds in [('OBJECT_RA',[41,42,43]),('GOAL_RA',[41,42,43]),('GOAL_Coreset',[41,42,43,44])] for s in seeds]
 if branch=='OBJECT_RA':
  p=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%seed);w=Q/('seed%d'%seed);a='ra_coreset';accept=p/'AUTONOMOUS_EXIT_ACCEPTANCE.json'
 elif branch=='GOAL_RA':
  p=Path('/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed%d_%s'%(seed,'20261018' if seed==42 else '20261019'))
  w=Path('/root/mlr_ifa/experiment_workspaces')/('RA_Coreset_GOAL_seed41_three_arm_20261019/ra' if seed==41 else 'RA_Coreset_GOAL_seed42_20261018' if seed==42 else 'RA_Coreset_GOAL_extension_20261019/seed43')
  a='ra_coreset';accept=p/'PROCESS_EXIT_ACCEPTANCE.json'
 else:
  p=Path('/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed%d_%s'%(seed,'20261019' if seed==41 else '20261017'))
  w=Path('/root/mlr_ifa/experiment_workspaces')/('RA_Coreset_GOAL_seed41_three_arm_20261019/baseline' if seed==41 else 'Coreset_paper_GOAL_three_seed_20261017/seed%d'%seed)
  a='coreset';accept=p/'INDEPENDENT_ACCEPTANCE.json'
 env=w/'env.sh'
 if not env.exists():env=w/'source/coreset_exp/env.sh'
 return dict(parent=str(p),parent_source=str(w/'source'),parent_env=str(env),parent_arm=a,parent_engineering_file=str(accept))
def prepare(branch,seed):
 from flow_support import verify_controller
 from flow_metrics import metrics,pack
 verify_controller();spec=parent_spec(branch,seed);P=Path(spec['parent']);S=Path(spec['parent_source']);arm=spec['parent_arm']
 assert json.loads(Path(spec['parent_engineering_file']).read_text())['status']=='PASS'
 assert S.is_dir() and Path(spec['parent_env']).is_file()
 counts=[1562,1480,1560,1319] if branch=='OBJECT_RA' else [997,889,904,1969]
 fm=json.loads((P/'FEATURES.json').read_text());assert [fm['counts'][str(t)] for t in range(6,10)]==counts
 m,rates,records,hs=metrics(P/arm/'evaluation',seed,arm)
 quotas=[(x+1)//2 for x in counts];lens=[v+sum(quotas[:i]) for i,v in enumerate(counts)]
 for t in range(6,10):
  cp=P/arm/'models'/('mlrifa_%d_model_ep99.pth'%t);h=hashlib.sha256(cp.read_bytes()).hexdigest()
  assert h==records[t,t,0]['checkpoint_sha256']
  st=json.loads((P/arm/'storage'/('task%d.json'%t)).read_text());assert st['K']==quotas[t-6]
  assert len(st['selected'])==len({x['destination'] for x in st['selected']})==quotas[t-6]
 W=N/'zr_runs'/('%s_seed%d'%(branch,seed));O=Path('/data/mlr_ifa/experiments/Conditional_ZRv2_%s_seed%d_20261019'%(branch,seed))
 assert not W.exists() and not O.exists(),'No overwrite/duplicate preparation'
 assert shutil.disk_usage('/data').free>2*1024**3,'Need 2GiB free before another seed; do not delete assets'
 shutil.copytree(N/'zr_template',W);O.mkdir()
 spec.update(seed=seed,branch=branch,suite='LIBERO_OBJECT' if branch=='OBJECT_RA' else 'LIBERO_GOAL',arm=arm+'_ifa_zrv2',output=str(O),counts=counts,quotas=quotas,epochs=40,optimizer_steps=10*sum((x+9)//10 for x in lens),exposures=10*sum(lens),formal_episodes=200,controller_contract=str(N/'AUTONOMOUS_CONTRACT.json'),baseline_metrics=pack(m),baseline_episode_hashes=hs,parent_source_contract_sha256=hashlib.sha256((P/'CONTRACT.json').read_bytes()).hexdigest(),additional_public_training=0,local_residual_seed='experiment_seed + task_id',evaluation='original 0-19/20 per cell/10 lanes/env[s,s+1]/600/5zero/stage global RNG/argmax component Normal',created=time.time())
 if branch=='OBJECT_RA':spec['parent_amendment_files']=[str(N/n) for n in ['RECOVERY_CONTRACT.json','AUTHORIZATION.json','ra_adapter.py','ra_audit.py','stage1/%d/PROTOCOL_EQUIVALENCE.json'%seed,'stage1/%d/TASK6_POLICY_ACCEPTANCE.json'%seed]]
 (W/'RUN_SPEC.json').write_text(json.dumps(spec,indent=2)+'\n')
 (W/'env.sh').write_text('#!/usr/bin/env bash\nsource '+spec['parent_env']+'\nexport PYTHONPATH="'+str(W/'source')+':'+str(W/'attachment')+':$PYTHONPATH"\nexport PYTHONDONTWRITEBYTECODE=1\ncd '+str(W)+'\n')
 return W,O,spec
if __name__=='__main__':
 w,o,s=prepare(sys.argv[1],int(sys.argv[2]));print(json.dumps(dict(workspace=str(w),output=str(o),spec=s)))
