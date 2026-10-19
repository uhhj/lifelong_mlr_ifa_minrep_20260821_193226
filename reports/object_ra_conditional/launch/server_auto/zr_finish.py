"""After pipeline child exits: independent record/attempt/hash/process acceptance."""
import sys,json,time
from pathlib import Path
from collections import Counter
from flow_support import js,sha,write,no_workers
from flow_metrics import metrics,pack
def main():
 W=Path(sys.argv[1]);spec=js(W/'RUN_SPEC.json');O=Path(spec['output']);P=Path(spec['parent']);seed=spec['seed'];arm=spec['arm'];pa=spec['parent_arm']
 assert (O/'pipeline.exit').read_text().strip()=='0';no_workers()
 for n in ['calibrate','evaluate','audit']:assert (O/(n+'.exit')).read_text().strip()=='0'
 a=js(O/'AUDIT.json');assert a['status']=='PASS' and a['epochs']==40 and a['optimizer_steps']==spec['optimizer_steps'] and a['sample_exposures']==spec['exposures']
 l=js(O/'CHECKPOINT_LOAD_ACCEPTANCE.json');assert l['status']=='PASS' and len(l['rows'])==4
 for n,h in js(O/'CONTRACT.json')['files'].items():assert sha(n)==h,n
 events=Counter(json.loads(x)['event'] for f in (O/'evaluation').glob('attempts*.jsonl') for x in f.read_text().splitlines())
 assert events==Counter(vector_attempt_start=20,vector_attempt_complete=20,environment_construct_start=10,environment_construct_complete=10,environment_closed=10)
 pm,pr,prows,ph=metrics(P/pa/'evaluation',seed,pa);nm,nr,nrows,nh=metrics(O/'evaluation',seed,arm)
 assert ph==spec['baseline_episode_hashes']
 for k in prows:
  assert prows[k]['checkpoint_sha256']==nrows[k]['parent_checkpoint_sha256']
  for field in ['initial_state_index','initial_state_sha256','modal_sha256']:assert prows[k][field]==nrows[k][field]
 assert pack(nm)==js(O/'METRIC_COMPARISON.json')['new']
 out=dict(status='PASS',time=time.time(),seed=seed,branch=spec['branch'],all_phases_exit=0,epochs=40,optimizer_steps=spec['optimizer_steps'],records=200,attempts=20,strict_checkpoints=4,baseline=pack(pm),new=pack(nm),differences=pack({k:nm[k]-pm[k] for k in nm}),parent_records_unchanged=True,formal_input_hashes='PASS',no_remaining_workers=True,new_episode_hashes=nh)
 write(O/'PROCESS_EXIT_ACCEPTANCE.json',out);write(O/'RUN_STATUS.json',dict(status='COMPLETED_PENDING_GROUP_GATE',time=time.time()))
 print(json.dumps(out))
if __name__=='__main__':main()
