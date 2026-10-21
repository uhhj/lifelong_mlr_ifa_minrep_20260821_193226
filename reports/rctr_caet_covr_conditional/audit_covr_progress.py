"""Read-only partial-stage accounting, never a final acceptance or efficacy claim."""
import argparse,collections,datetime,hashlib,json,statistics
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();p=a.directory
 raw=(p/'COVR_DETAILS.json').read_bytes();d=json.loads(raw)
 assert hashlib.sha256(raw).hexdigest()==json.loads((p/'COVR_DETAILS_RECEIPT.json').read_text())['sha256']
 assert d['controller_exit'] is None and d['blocked'] is None and d['contract']['hashes']=='PASS'
 assert d['initialization_equal'] and all(d['ifa_equal'].values()) and d['task6_tensor']['equal']
 rows=d['epochs'];counts={6:157,7:227,8:309,9:362};total=0;seen=set()
 for x in rows:
  key=(x['task'],x['epoch']);assert key not in seen and 0<=x['epoch']<100;seen.add(key)
  assert x['batches']==counts[x['task']];total+=x['batches'];assert total==x['steps']
 assert all(v=='0' for v in d['exits'].values())
 starts=collections.Counter(x['phase'] for x in d['phase_events'] if x['event']=='start');assert all(n==1 for n in starts.values())
 reports={}
 for name,c in d['collections'].items():
  stage=int(name[5:]);br=c['branches'];events=c['events'];keys=[r['key'] for r in br];assert len(keys)==len(set(keys))
  bs=[x['key'] for x in events if x['event']=='branch_start'];be=[x['key'] for x in events if x['event']=='branch_complete']
  assert len(bs)==len(set(bs)) and len(be)==len(set(be)) and set(keys)<=set(be)<=set(bs)
  assert len(set(bs)-set(be))<=1 and len(set(be)-set(keys))<=1
  exp=sum(r['phase']=='exploration' for r in br);conf=len(br)-exp
  assert exp<=54*(stage-6) and conf<=27*(stage-6)
  noise={}
  for x in br:
   assert x['stage']==stage and 6<=x['task']<stage and x['initial'] in range(20,26)
   at=(40,120,240,40,120,240)[x['initial']-20]
   assert x['snapshot_step']==at and 0<x['steps']<=600-at and type(x['success']) is bool
   assert x['repeat'] in range(3) and x['mode'] in ['normal',0,1,2,3,4]
   expected=42*100000000+stage*1000000+x['task']*10000+x['initial']*100+x['repeat']+(500000 if x['phase']=='confirmation' else 0)
   assert x['noise_seed']==expected
   k=(x['task'],x['initial'],x['phase'],x['repeat']);assert noise.setdefault(k,x['noise_sha256'])==x['noise_sha256']
  assert c['payload_hashes']=='PASS' and all(g['status']=='PASS' for g in c['gates'])
  ns=[(n['task'],n['initial']) for n in c['normal']];assert len(ns)==len(set(ns)) and len(ns)<=6*(stage-6)
  assert all(n['stage']==stage and 0<n['steps']<=600 for n in c['normal'])
  assert not any(x['event'].startswith('technical_') for x in events), 'Technical attempts require explicit separate audit'
  avg=statistics.mean(x['seconds'] for x in br) if br else None
  reports[name]=dict(committed=len(br),exploration=exp,confirmation=conf,normal_completed=len(ns),
   successful_branches=sum(x['success'] for x in br),policy_steps=sum(x['steps'] for x in br),
   completed_decisions=len(c['decisions']),acceptance=c['acceptance'],restore_gates=len(c['gates']),
   mean_branch_seconds=avg,remaining_exploration_budget_minutes=(54*(stage-6)-exp)*avg/60 if avg else None,
   remaining_confirmation_budget_minutes=(27*(stage-6)-conf)*avg/60 if avg else None,
   note='Budget-scaled projection, not all branches necessarily executed; excludes normal episodes and engineering overhead')
 out=dict(status='COVR_PROGRESS_PASS',experiment_status='PARTIAL',
  observed_at=datetime.datetime.fromtimestamp(d['observed_end_unix'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
  epochs=len(rows),optimizer_steps=total,phase=d['run_status'],auxiliary=d['auxiliary'],collections=reports,
  full_run_eta=None,source_input_hashes='PASS',storage=d['storage'],task6_tensor_equal=True,initialization_equal=True,
  local_optimizer_calls=0,local_environment_steps=0,details_sha256=hashlib.sha256(raw).hexdigest())
 (p/'COVR_PROGRESS_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
