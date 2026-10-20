"""Independent exact arithmetic on downloaded outcomes; zero ML/environment use."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,zipfile,statistics,datetime
R=Path(__file__).resolve().parent;C=R/'completion_20261020';D=C/'download'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=js(C/'completion_download_receipt.json');zpath=C/'completion_reviewed_20261020.zip'
assert sha(zpath)==receipt['sha256'] and zpath.stat().st_size==receipt['bytes']
with zipfile.ZipFile(zpath) as z:
 manifest=json.loads(z.read('DOWNLOAD_MANIFEST.json'))
 assert set(z.namelist())==set(manifest['files'])|{'DOWNLOAD_MANIFEST.json'}
 for n in z.namelist():
  p=Path(n);assert not p.is_absolute() and '..' not in p.parts
  assert p.suffix in ['.json','.jsonl','.md','.log','.exit']
  raw=z.read(n)
  if n!='DOWNLOAD_MANIFEST.json':
   v=manifest['files'][n];assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256']
  dest=D/p;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert dest.read_bytes()==raw
  else:dest.write_bytes(raw)
for n,v in manifest['files'].items():assert sha(D/n)==v['sha256']
supp=js(C/'stage1_supplement_receipt.json');suppzip=C/'stage1_metadata_supplement_20261020.zip'
assert sha(suppzip)==supp['sha256'] and suppzip.stat().st_size==supp['bytes']
with zipfile.ZipFile(suppzip) as z:
 sm=json.loads(z.read('SUPPLEMENT_MANIFEST.json'));assert set(z.namelist())==set(sm['files'])|{'SUPPLEMENT_MANIFEST.json'}
 for n in z.namelist():
  assert not Path(n).is_absolute() and '..' not in Path(n).parts and Path(n).suffix=='.json'
  raw=z.read(n)
  if n!='SUPPLEMENT_MANIFEST.json':assert hashlib.sha256(raw).hexdigest()==sm['files'][n]['sha256'] and len(raw)==sm['files'][n]['bytes']
  dest=D/'supplement'/n;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert dest.read_bytes()==raw
  else:dest.write_bytes(raw)
for n in ['controller','wrapper','supervisor','final_audit','package_light']:assert (D/'current'/(n+'.exit')).read_text().strip()=='0'
snap=js(D/'current/DEPLOYMENT_PROGRESS.json')
assert not snap['workers'] and all(not v['alive'] for v in snap['processes'].values())
assert snap['frozen_controller_hashes']=='PASS'
final=js(D/'current/FINAL_SERVER_ACCEPTANCE.json');results=js(D/'current/EXPERIMENT_RESULTS.json')
assert final['status']=='PASS' and final['formal_records']==1200 and final['strict_checkpoints']==24
assert results['executed']==['OBJECT_RA_IFA','OBJECT_RA_ZRv2_IFA']
assert results['results']['stop_reason']=='STAGE2_METRIC_GATE_NOT_MET'
assert not {'stage3','fallback'}&set(results['results'])

def outcomes(folder,seed,arm):
 cells={};records={};hashes={}
 files=list(folder.glob('trials*.jsonl'));assert len(files)==10
 for f in files:
  rows=[json.loads(l) for l in f.read_text().splitlines()];assert len(rows)==20
  s,t=rows[0]['stage'],rows[0]['task'];assert 6<=t<=s<=9 and (s,t) not in cells
  assert {x['trial'] for x in rows}==set(range(20))
  for x in rows:
   assert x['stage']==s and x['task']==t and x['seed']==seed and x['arm']==arm
   assert type(x['success']) is bool and x['initial_state_index']==x['trial']
   records[s,t,x['trial']]=x
  cells[s,t]=F(sum(x['success'] for x in rows),20);hashes[f.name]=sha(f)
 assert len(records)==200
 # Compute per-task learning-curve means and forgetting; no server metric function imported.
 curves={t:[cells[s,t] for s in range(t,10)] for t in range(6,10)}
 auc=sum(sum(v)/len(v) for v in curves.values())/4
 forgetting=[sum(v[0]-x for x in v[1:])/(len(v)-1) for v in curves.values() if len(v)>1]
 m=dict(AUC=auc,FWT=sum(v[0] for v in curves.values())/4,NBT=sum(forgetting)/3,
        OldFinal=sum(cells[9,t] for t in (6,7,8))/3,FinalAll=sum(cells[9,t] for t in range(6,10))/4)
 return m,cells,records,hashes
def pack(m):return {k:dict(fraction=str(v),percent=float(v*100)) for k,v in m.items()}
def events(folder):
 counts=Counter(json.loads(l)['event'] for f in folder.glob('attempts*.jsonl') for l in f.read_text().splitlines())
 assert counts==Counter(vector_attempt_start=20,vector_attempt_complete=20,environment_construct_start=10,environment_construct_complete=10,environment_closed=10)
 return dict(counts)
data={};full={};training_seconds={};new_records=0;attempts=0
for seed in [41,42,43]:
 ra=D/'evidence'/('OBJECT_RA_seed%d'%seed);zr=D/'evidence'/('OBJECT_RA_seed%d'%seed)
 # Package prefixes for RA and ZRv2 are distinct: inspect spec-bound ZRv2 prefix.
 spec=js(D/'specs'/('OBJECT_RA_seed%d'%seed)/'RUN_SPEC.json')
 zr=D/'evidence'/('OBJECT_RA_seed%d'%seed)
 # Both collectors used the same prefix for OBJECT_RA stage1 and stage2!
 # Distinguish actual phase files: stage1 lives under ra_coreset/, stage2 at root.
 assert (ra/'ra_coreset/epochs.json').exists() and (zr/'epochs.json').exists()
 roots={'random':D/'baselines'/('seed%d'%seed)/'random/evaluation',
        'coreset':D/'baselines'/('seed%d'%seed)/'coreset/evaluation',
        'ra_coreset':ra/'ra_coreset/evaluation','ra_coreset_ifa_zrv2':zr/'evaluation'}
 data[str(seed)]={};full[seed]={}
 for arm,folder in roots.items():
  m,c,rows,hs=outcomes(folder,seed,arm);full[seed][arm]=(m,c,rows,hs)
  ref=results['results']['stage1']['seed_results'][str(seed)][arm]['metrics'] if arm!='ra_coreset_ifa_zrv2' else results['results']['stage2']['seed_results'][str(seed)]['new']
  assert pack(m)==ref
  data[str(seed)][arm]=dict(metrics=pack(m),matrix=[[float(c[s,t]*100) if t<=s else None for t in range(6,10)] for s in range(6,10)],hashes=hs)
 for arm in ['ra_coreset','ra_coreset_ifa_zrv2']:
  events(roots[arm]);new_records+=200;attempts+=20
 epochs=js(ra/'ra_coreset/epochs.json')
 assert len(epochs)==400 and {(v['task'],v['epoch']) for v in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
 assert sum(v['batches'] for v in epochs)==105500
 cal=js(zr/'epochs.json')
 assert len(cal)==40 and {(v['task'],v['epoch']) for v in cal}=={(t,e) for t in range(6,10) for e in range(10)}
 assert sum(v['batches'] for v in cal)==10550 and sum(v['samples'] for v in cal)==105240
 training_seconds[str(seed)]=dict(ra_committed_epochs=sum(v['seconds'] for v in epochs),zr_calibration_epochs=sum(v['seconds'] for v in cal))
 engineering=js(D/'evidence/controller/stage1'/str(seed)/'ENGINEERING_ACCEPTANCE.json')
 assert engineering['status']=='PASS' and engineering['physical_steps']==105500+(157 if seed==41 else 0)
 ra_audit=js(D/'supplement'/('RA_STAGE1_seed%d'%seed)/'AUDIT.json')
 assert ra_audit['status']=='PASS' and ra_audit['inputs_unchanged'] and len(ra_audit['checkpoints'])==4
 assert all(x['loaded_keys']==100 for x in ra_audit['checkpoints'])
 assert ra_audit['metrics']=={a:pack(full[seed][a][0]) for a in ['random','coreset','ra_coreset']}
 for row in full[seed]['ra_coreset'][2].values():assert row['checkpoint_sha256']==ra_audit['checkpoints'][row['stage']-6]['sha256']
 for t in range(6,10):
  store=js(ra/'ra_coreset/storage'/('task%d.json'%t));offline=js(ra/'offline_selection'/('task%d.json'%t))
  assert store['selection']['selected_ids']==offline['selected_ids']
  assert len(store['selected'])==len(set(store['selection']['selected_ids']))==[781,740,780,660][t-6]
  assert set(store['selection']['anchor_ids'])<=set(store['selection']['selected_ids'])
  assert store['selection']['swap_count']==len(set(store['selection']['anchor_ids'])-set(store['selection']['original_ids']))
 assert js(zr/'ADMISSION.json')['status']=='PASS' and js(zr/'INTEGRATION.json')['status']=='PASS'
 assert js(zr/'INTEGRATION.json')['optimizer_calls']==4 and js(zr/'INTEGRATION.json')['environment_steps']==0
 structural=[json.loads(l) for l in (zr/'STRUCTURE.jsonl').read_text().splitlines()]
 assert len(structural)==16 and all(v['passed'] for v in structural)
 cp=js(zr/'CHECKPOINT_LOAD_ACCEPTANCE.json');assert cp['status']=='PASS' and len(cp['rows'])==4
 zra=js(zr/'PROCESS_EXIT_ACCEPTANCE.json');assert zra['status']=='PASS' and zra['records']==200
 for key,row in full[seed]['ra_coreset_ifa_zrv2'][2].items():
  par=full[seed]['ra_coreset'][2][key]
  assert row['parent_checkpoint_sha256']==par['checkpoint_sha256']
  for field in ['initial_state_index','initial_state_sha256','modal_sha256']:assert row[field]==par[field]
  assert row['checkpoint_sha256']==cp['rows'][row['stage']-6]['checkpoint_sha256']
 # Parent RA membership and all copied payload hashes are unchanged by ZRv2.
 for t in range(6,10):
  assert js(zr/('buffer_after_task%d.json'%t))['hashes']==js(ra/'ra_coreset'/('buffer_after_task%d.json'%t))['hashes']
assert new_records==1200 and attempts==120
gates={}
for phase,a,b in [('stage1','ra_coreset','random'),('stage2','ra_coreset_ifa_zrv2','ra_coreset')]:
 ds=[full[s][a][0]['AUC']-full[s][b][0]['AUC'] for s in [41,42,43]]
 mean=sum(ds)/3;count=sum(d>0 for d in ds);passed=mean>0 and count>=2
 ref=results['results'][phase]['gate']
 assert (str(mean),count,passed)==(ref['mean_auc_delta_fraction'],ref['positive_seeds'],ref['passed'])
 gates[phase]=dict(passed=passed,mean_auc_delta_pp=float(mean*100),mean_fraction=str(mean),positive_seeds=count,paired_delta_pp=[float(d*100) for d in ds],paired_sample_SD_pp=statistics.stdev(float(d*100) for d in ds))
assert gates['stage1']['passed'] and not gates['stage2']['passed']
stats={}
for arm in full[41]:
 stats[arm]={}
 for k in full[41][arm][0]:
  vs=[full[s][arm][0][k] for s in [41,42,43]]
  stats[arm][k]=dict(mean_fraction=str(sum(vs)/3),mean_percent=float(sum(vs)/3*100),sample_SD_percent=statistics.stdev(float(v*100) for v in vs))
out=dict(status='PASS',time=datetime.datetime.now().astimezone().isoformat(),downloaded_files=len(manifest['files']),seed_results=data,statistics=stats,gates=gates,formal_new_records=new_records,reused_baseline_records=1200,vector_attempts=attempts,strict_checkpoint_evidence=24,base_epochs=1200,calibration_epochs=120,logical_optimizer_steps=348150,extra_physical_replay_steps=157,epoch_seconds=training_seconds,server_exit_codes=0,no_remaining_workers=True,next_action='STOP_STAGE2_GATE_FAILED_NO_GOAL',final_acceptance='COMPLETED')
out['supplement_metadata_files']=len(sm['files']);out['package_collision_fixed_readonly']=True
(C/'LOCAL_FINAL_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],gates=gates,statistics=stats,epoch_seconds=training_seconds),indent=2))
