"""Finite server exit acceptance, independent count recomputation, no policy calls."""
import sys,time,json
from pathlib import Path
from flow_support import N,js,write,verify_controller,no_workers,alive
from flow_metrics import metrics,pack
def main():
 assert (N/'controller.exit').read_text().strip()=='0'
 assert (N/'wrapper.exit').read_text().strip()=='0'
 assert not alive(js(N/'controller.started')['pid']);no_workers();verify_controller()
 r=js(N/'EXPERIMENT_RESULTS.json');s1=r['results']['stage1'];assert s1['status']=='PASS'
 total_records=600;total_cp=12;epochs=1200;steps=316500
 for key in ['stage2','stage3','fallback']:
  if key not in r['results']:continue
  g=r['results'][key];assert g['status']=='PASS'
  for seed,row in g['seed_results'].items():
   O=Path(row['output']);spec_candidates=list((N/'zr_runs').glob(g['branch']+'_seed'+seed+'/RUN_SPEC.json'));assert len(spec_candidates)==1
   spec=js(spec_candidates[0]);m,_,_,_=metrics(O/'evaluation',int(seed),spec['arm']);assert pack(m)==row['new']
   assert js(O/'PROCESS_EXIT_ACCEPTANCE.json')['status']=='PASS'
  total_records+=g['formal_records'];total_cp+=g['strict_checkpoints'];epochs+=g['calibration_epochs'];steps+=g['optimizer_steps']
 if s1['gate']['passed']:
  assert 'stage2' in r['results'] and 'fallback' not in r['results']
  assert ('stage3' in r['results'])==r['results']['stage2']['gate']['passed']
 else:assert 'fallback' in r['results'] and 'stage2' not in r['results'] and 'stage3' not in r['results']
 write(N/'FINAL_SERVER_ACCEPTANCE.json',dict(status='PASS',time=time.time(),executed=r['executed'],formal_records=total_records,vector_attempts=total_records//10,strict_checkpoints=total_cp,base_training_epochs=1200,calibration_epochs=epochs-1200,logical_optimizer_steps=steps,extra_replayed_physical_steps=157,controller_exit=0,wrapper_exit=0,no_remaining_workers=True,all_engineering_checks=True,local_download_publication='pending desktop availability'))
 write(N/'STATUS.json',dict(status='COMPLETED_SERVER_PENDING_LOCAL_PUBLICATION',time=time.time(),formal_records=total_records,executed=r['executed']))
 lines=['# Conditional RA / ZRv2 study — server final report','', 'All authorized executed branches passed engineering acceptance. See FINAL_SERVER_ACCEPTANCE.json.','']
 for key,g in r['results'].items():
  if not isinstance(g,dict):lines+=['Stop reason: '+str(g)];continue
  lines+=['## '+key,'', '|Seed|Method|AUC %|FWT %|NBT pp|OldFinal %|FinalAll %|','|---|---|---:|---:|---:|---:|---:|']
  for seed,row in g['seed_results'].items():
   entries={a:v['metrics'] for a,v in row.items()} if key=='stage1' else {a:row[a] for a in ['parent','new']}
   for arm,m in entries.items():lines+=['|'+seed+'|'+arm+'|'+'|'.join('%.8f'%m[k]['percent'] for k in ['AUC','FWT','NBT','OldFinal','FinalAll'])+'|']
  lines+=['','All task matrices, exact fractions and sample SD (ddof=1): EXPERIMENT_RESULTS.json.']
  if 'gate' in g:lines+=['Gate: '+json.dumps(g['gate'])]
 lines+=['','Exploratory conditional sequence; historical hardware differs and episode action RNG is not isolated. Pretrain state metadata remains unknown. IFA pair choices may change indirectly with memory. ZRv2 incurs additional calibration and a fixed T0 inference branch. Task6 cross-hardware tensor differences are disclosed under the explicit user amendment; initialization and protocol checks remain mandatory. No significance or universal causal claim.']
 (N/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(js(N/'FINAL_SERVER_ACCEPTANCE.json')))
if __name__=='__main__':main()
