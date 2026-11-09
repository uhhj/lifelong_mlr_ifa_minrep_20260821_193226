from pathlib import Path
import json,ast,hashlib,difflib
R=Path(__file__).resolve().parent;E=R.parent
assert not (R/'CONFIGURATION_REVIEW.json').exists(),'No repeated configuration'
roots=['/data/mlr_ifa/experiments/S_Q1_'+s+'41_42_43_20261109_rerun_v1' for s in ['OBJECT','GOAL']]
review={}
def read(p):return p.read_text(encoding='utf-8-sig')
def put(p,s):p.write_text(s,encoding='utf8',newline='\n')
def rep(s,a,b):assert a in s,a;return s.replace(a,b)
for suite in ['OBJECT','GOAL']:
 S=R/suite
 for seed in [41,42,43]:
  q=S/f'Q1_{seed}'
  p=q/'layout.py';s=read(p);s=rep(s,"'/data':4*1024**3","'/data':6*1024**3");s+='\nCAPACITY_OUTPUT_ROOTS='+repr(roots)+'\n';put(p,s)
  p=q/'runtime.py';s=read(p);s=rep(s,"for p in base.rglob('*') if base.exists() else []:","for p in (p for b in ([Path(x) for x in CAPACITY_OUTPUT_ROOTS] if mount=='/data' else [base]) for p in (b.rglob('*') if b.exists() else [])):");put(p,s)
 p=S/'check_readonly.py';s=read(p);s=rep(s,"for p in PRIVATE.rglob('*') if PRIVATE.exists() else []:","for p in (p for b in map(Path,"+repr(roots)+") for p in (b.rglob('*') if b.exists() else [])):");s=rep(s,"('/data',4*1024**3","('/data',6*1024**3");put(p,s)
 for n in ['audit_final_local.py','compare_primary_s.py']:
  p=S/n;s=read(p).replace('Fresh '+suite+'3090','Repeated '+suite+'4080SUPER').replace('USER_AUTHORIZED_RTX3090','USER_AUTHORIZED_Q1_RERUN_OBSERVED_RTX4080_SUPER');put(p,s)
 p=S/'final_readonly.py';s=read(p);s=s[s.index('"""Only after natural completion:'):]
 scripts={f'Q1_{seed}':read(S/f'final_Q1_{seed}_readonly.py') for seed in [41,42,43]}
 put(p,'CHECK_SCRIPT='+repr(read(S/'check_readonly.py'))+'\nSCRIPTS='+repr(scripts)+'\n'+s)
 old=E/f'20261108_{suite.lower()}_s_q1_ablation'
 for p in S.rglob('*.py'):
  s=read(p);ast.parse(s,feature_version=(3,8));rel=p.relative_to(S);base=old/rel
  if base.exists():
   before=read(base);review[suite+'/'+rel.as_posix()]=dict(old_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),unchanged=before==s)
   if before!=s:
    dest=R/'review'/suite/rel.with_suffix('.diff');dest.parent.mkdir(parents=True,exist_ok=True);put(dest,''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile=str(base),tofile=str(p))))
 # No scientific algorithm change: exact accepted q1 source reused for every loss/update/collection/evaluation.
 for seed in [41,42,43]:
  for n in ['support_memory.py','losses.py','legacy_losses.py','updates.py','legacy_updates.py','candidate.py','collect.py','evaluate.py','gate_admission.py','train.py','memory.py','legacy_memory.py']:
   assert read(S/f'Q1_{seed}'/n)==read(old/f'Q1_{seed}'/n).replace(f'S_Q1_{suite}41_42_43_20261108_v1',f'S_Q1_{suite}41_42_43_20261109_rerun_v1'),(suite,seed,n)
status=dict(status='PASS_STATIC_REUSE_AND_ENGINEERING_DIFF_REVIEW_NOT_DEPLOYED',files=review,science_change_from_previous_q1='NONE',allowed_changes=['new paths/identities','actual GPU guard 4080 SUPER and truthful hardware metadata','combined six-run capacity accounting; same reserves','sequential suite coordinator'],capacity=dict(formula='6*(3*90MiB+360MiB+4*10MiB)+2*360MiB+1GiB',calculated_bytes=6*(3*90+360+4*10)*1024**2+2*360*1024**2+1024**3,rounded_data_bound=6*1024**3,data_reserve=2*1024**3,root_bound=128*1024**2,root_reserve=1024**3,final_stage_is_hardlink_to_recovery=True),final_wrappers='STATIC_ONLY_MUST_ACTUALLY_PASS_AFTER_NATURAL_COMPLETION')
put(R/'CONFIGURATION_REVIEW.json',json.dumps(status,indent=2)+'\n');print(json.dumps(dict(status=status['status'],reviewed_files=len(review),capacity=status['capacity']),indent=2))
