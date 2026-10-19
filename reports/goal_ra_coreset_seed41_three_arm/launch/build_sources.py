"""Prepare new seed41 namespaces; preserve all scientific algorithms byte-for-byte."""
from pathlib import Path
import shutil,json,hashlib,difflib,yaml
R=Path(__file__).resolve().parent;X=R.parent/'20261019_ra_coreset_goal_extension';B=R.parent/'20261017_coreset_goal_three_seed/seed42'
Q='/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed41_three_arm_20261019'
BO='/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019';RO='/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed41_20261019'
proof={};patch=[]
for arm,src in [('baseline',B/'source'),('ra',X/'source')]:
 D=R/arm/'source'
 if D.exists():assert (R/'PREPARATION_REPAIR.json').exists() and not (R/'LAUNCH.json').exists()
 shutil.copytree(src,D,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
 def edit(n,old,new):
  p=D/n;s=p.read_text(encoding='utf-8');assert old in s,(arm,n,old);p.write_text(s.replace(old,new),encoding='utf-8',newline='\n')
 edit('coreset_exp/common.py','assert SEED in (42,43,44)','assert SEED == 41')
 if arm=='baseline':
  oldW='/root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/seed42/source';newW=Q+'/baseline/source'
  for n in ['env.sh','pipeline.sh','supervise.sh']:edit('coreset_exp/'+n,oldW,newW)
  edit('coreset_exp/env.sh','Coreset_paper_GOAL_seed42_20261017','Coreset_paper_GOAL_seed41_20261019')
  edit('coreset_exp/env.sh','CORESET_SEED=42','CORESET_SEED=41')
  edit('coreset_exp/pipeline.sh','39342','39441')
  edit('coreset_exp/configure.py','extensions=[42,43,44]','extensions=[41]')
  edit('coreset_exp/configure.py','paper_aligned_GOAL_three_seed_v1','paper_aligned_GOAL_seed41_three_arm_v1')
  edit('coreset_exp/configure.py','User authorized GOAL Random/Coreset seeds42/43/44; system disk outputs; fresh paired baselines due disclosed historical differences','User explicitly authorized GOAL seed41 Random/Coreset/RA three-arm experiment; no existing GOAL41 baseline; unchanged protocol and system-disk outputs')
  old="for root in ('/data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/run', *[f'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{s}_20261013' for s in (42,43,44)]):"
  new="for root in ('/data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/run', *[f'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{s}_20261013' for s in (42,43,44)], *[f'/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed{s}_20261017' for s in (42,43,44)], '/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018', *[f'/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed{s}_20261019' for s in (43,44)]):"
  edit('coreset_exp/configure.py',old,new)
  edit('coreset_exp/summarize.py','extensions=[42,43,44]','extensions=[41]')
  edit('coreset_exp/summarize.py','仅执行用户授权的GOAL三seed42/43/44','仅执行用户授权的GOAL seed41新增两组配对基线')
 else:
  edit('ra_exp/prepare.py','SEED in (43,44)','SEED == 41')
  W=Q+'/ra'
  for n in ['env.sh','admission.sh','pipeline.py','supervise.sh','final_process_audit.py','package_light.py']:
   s=(X/'seed43'/n).read_text(encoding='utf-8').replace('/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_extension_20261019/seed43',W).replace('CORESET_SEED=43','CORESET_SEED=41').replace('/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed43_20261019',RO).replace('/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed43_20261017',BO).replace('/root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/seed43/source',Q+'/baseline/source')
   (R/'ra'/n).write_text(s,encoding='utf-8',newline='\n')
 changed={};same={}
 for p in sorted(D.rglob('*')):
  if not p.is_file():continue
  n=p.relative_to(D).as_posix();h=hashlib.sha256(p.read_bytes()).hexdigest()
  if p.read_bytes()==(src/n).read_bytes():same[n]=h
  else:
   changed[n]=h;patch.extend(difflib.unified_diff((src/n).read_text(encoding='utf-8').splitlines(True),p.read_text(encoding='utf-8').splitlines(True),fromfile=arm+'_parent/'+n,tofile=arm+'_41/'+n))
  if p.suffix=='.py':compile(p.read_text(encoding='utf-8'),str(p),'exec')
 frozen=['coreset_exp/train.py','coreset_exp/evaluate.py','coreset_exp/eval_metric.py','coreset_exp/model.py','coreset_exp/storage.py','libero/lifelong/selectors/greedy_kcenter.py','libero/lifelong/models/brief.py','libero/lifelong/models/complete_policy.py']
 if arm=='ra':frozen+=['ra_exp/selector_reference.py','ra_exp/select_offline.py','ra_exp/preflight.py']
 for n in frozen:assert (D/n).read_bytes()==(src/n).read_bytes(),n
 proof[arm]=dict(parent=str(src),changed=changed,unchanged=same,scientific_core_unchanged=True)
 (R/arm/'SOURCE_SHA256.json').write_text(json.dumps({**same,**changed},indent=2)+'\n')
(R/'SOURCE_DIFF.json').write_text(json.dumps(proof,indent=2)+'\n');(R/'source.patch').write_text(''.join(patch),encoding='utf-8',newline='\n')
shutil.copyfile(R.parent/'20261017_coreset_goal_three_seed/audit_suite.py',R/'audit_suite.py')
spec=dict(status='PREPARING',seed=41,suite='LIBERO_GOAL',arms=['random','coreset','ra_coreset'],baseline_status='No existing GOAL41 baseline; explicit user authorization to create both',official_pretrain='/data/mlr_ifa/checkpoints/goal_multitask_model_ep96.pth',use_state=True,tasks=[6,7,8,9],demos_per_task=10,seq_len=8,batch=10,epochs_per_task=100,steps_per_arm=76100,total_epochs=1200,total_steps=228300,final_checkpoints=12,formal_episodes=600,vector_attempts=60,gamma=.7,ifa_weight=.1,scheduler='optimizer-step linear warmup500',sampler_seed=0,selector_seed=42,anchor_seed=41,quotas=[499,445,452,985],anchors=[249,222,226,492],algorithm='same RA hard random anchors and minimal geometric exchanges',evaluation=dict(initial_states='0..19',n_per_cell=20,lanes=10,max_steps=600,zero_steps=5,global_rng_reset_per_stage=41,environment_seeds=[41,42],action='argmax component then Normal; shared global RNG'),preflight=dict(baseline_software=18,baseline_optimizer_calls=4,baseline_zero_action_interface=1,baseline_policy_rollouts=0,RA_software=8,RA_optimizer_calls=4,RA_environment_calls=0),baseline_out=BO,ra_out=RO,queue=Q,auto_extend=False)
(R/'ACTUAL_SPEC.yaml').write_text(yaml.safe_dump(spec,sort_keys=False),encoding='utf-8')
print(json.dumps({k:list(v['changed']) for k,v in proof.items()},indent=2))
