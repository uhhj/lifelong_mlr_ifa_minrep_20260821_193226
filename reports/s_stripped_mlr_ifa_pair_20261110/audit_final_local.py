from pathlib import Path
import sys,json
from audit_progress import audit,read
d=Path(sys.argv[1]);a=audit(d);e=read(d/'EVIDENCE.json')
assert e['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT'
assert a['controller_exit']=='0' and not a['processes']
assert a['totals']==dict(epochs=4800,production=1089600,admission=240,formal=2400,training_collection=0)
assert a['complete_environment_intervals']==12 and a['environment_lock_held'] is None
assert set(e['cpu'])==set(a['cases'])
for case,x in e['cpu'].items():assert x['status']=='PASS_CPU_CASE_RECONSTRUCTION'
out=dict(status='PASS_COMPLETE_LOCAL_ACCEPTANCE',contract=a['contract'],observed=e['observed'],runs=12,checkpoints=48,recoveries=12,formal=2400,production=1089600,admission=240,training_collection=0,final_cpu_new_optimizer=0,final_cpu_new_environment=0)
(d/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out))
