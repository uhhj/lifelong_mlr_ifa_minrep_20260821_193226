"""Remote independent private-history verification; emits scalar/hash evidence only."""
import os,json,hashlib,time,subprocess
from pathlib import Path
import numpy as np
Q=Path(Q)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for v in iter(lambda:f.read(8*1024**2),b''):h.update(v)
 return h.hexdigest()
def ha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def js(p):return json.loads(p.read_text())
assert (Q/'worker.exit').read_text().strip()==(Q/'supervisor.exit').read_text().strip()=='0'
procs=subprocess.check_output(['ps','-eo','args'],text=True)
assert not any(str(Q)+'/paired_worker.py' in s or str(Q)+'/supervisor.py' in s for s in procs.splitlines())
ct=js(Q/'CONTRACT.json');assert all(sha(Q/n)==h for n,h in ct['files'].items())
spec=js(Q/'INPUT_SPEC.json');private=Path(ct['private_root']);rec=js(Q/'OUTCOMES.json');assert len(rec)==240
seen=set();checked=[];count=0
for r in rec:
 key=(r['seed'],r['method'],r['driver'],r['initial_index']);assert key not in seen;seen.add(key)
 run=r['run'];i=r['initial_index'];L=Q/'light'/run;P=private/run
 manifest=js(L/'PRIVATE_MANIFEST.json');assert all(sha(P/n)==v['sha256'] and (P/n).stat().st_size==v['bytes'] for n,v in manifest.items())
 rows_path=L/('trajectory_%02d.jsonl'%i);assert sha(rows_path)==r['rows_sha256'];rows=[json.loads(x) for x in rows_path.read_text().splitlines()]
 d=np.load(P/('%02d_history.npz'%i));frames=d['frames'];parameters=d['parameters'];actions=d['executed_actions'];assert len(rows)==len(frames)==r['policy_steps']
 eps=np.random.RandomState(20261024+r['seed']*1000+800+i).standard_normal((600,7)).astype(np.float32)
 assert ha(eps)==r['noise_sequence_sha256'] and np.array_equal(np.load(P/'noise.npy')[i%10],eps)
 event_first={};maxerr=0
 for t,x in enumerate(rows):
  assert x['step']==t and x['initial_index']==i and x['history_length']==min(t+1,8)
  assert x['history_sha256']==ha(frames[max(0,t-7):t+1]) and x['noise_sha256']==ha(eps[t])
  v=parameters[t];q=[]
  for off in [0,75]:
   mu=v[off:off+35].reshape(5,7);sd=v[off+35:off+70].reshape(5,7);logits=v[off+70:off+75];k=int(logits.argmax());q.append((mu[k],sd[k],k))
  (m,s,k),(m1,s1,k1)=q;dm=m1.astype(float)-m;ds=s1.astype(float)-s
  assert k==x['teacher_mode'] and k1==x['student_mode'] and x['mode_switch']==(k!=k1)
  assert np.allclose(dm**2,x['mean_delta_squared'],atol=1e-12,rtol=1e-12) and np.allclose(ds**2,x['sigma_delta_squared'],atol=1e-12,rtol=1e-12)
  a=m+s*eps[t];a1=m1+s1*eps[t];chosen=a if r['driver']==8 else a1
  maxerr=max(maxerr,float(np.max(np.abs(chosen-actions[t]))));assert np.allclose(chosen,actions[t],atol=2e-7,rtol=1e-6)
  flags={'mean_delta':float(np.abs(dm).max())>1e-3,'mode_switch':k!=k1,'gripper_flip':bool(m[6]*m1[6]<0)}
  for n,b in flags.items():
   if b and n not in event_first:event_first[n]=t
  count+=1
 assert event_first==r['first_events']
 for name in ['terminal']+['first_'+k for k in event_first]:
  path=P/('%02d_%s.npz'%(i,name));assert path.name in manifest
  raw=np.load(path);assert set(raw.files)=={'agentview_image','robot0_eye_in_hand_image','robot0_gripper_qpos','robot0_joint_pos'}
 checked.append({'run':run,'initial_index':i,'steps':len(rows),'history_reconstruction':'EXACT','executed_action_max_abs_recompute':maxerr,'private_history_sha256':manifest['%02d_history.npz'%i]['sha256']})
for e in spec['seeds']:
 for method,out,key in [('RA',e['ra_output'],'ra_checkpoints'),('ALIGN',e['output'],'checkpoints')]:
  O=Path(out);assert sha(O/'CONTRACT.json')==e['ra_contract_sha256' if method=='RA' else 'contract_sha256']
  for stage,h in e[key].items():assert sha(O/'ra_coreset/models'/('mlrifa_%s_model_ep99.pth'%stage))==h
 c=js(Path(e['output'])/'CONTRACT.json');assert all(sha(Path(e['source'])/n)==h for n,h in c['source_sha256'].items())
light={str(p.relative_to(Q)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (Q/'light').rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl']}
for n in ['ADMISSION.json','OUTCOMES.json','COMPLETE.json','EVENTS.jsonl','CONTRACT.json','worker.exit','supervisor.exit']:light[n]={'bytes':(Q/n).stat().st_size,'sha256':sha(Q/n)}
print(json.dumps({'status':'PASS_PRIVATE_HISTORY_ACTION_RECOMPUTATION','time':time.time(),'episodes':len(rec),'steps':count,'checks':checked,'light_files':light,'source_cp_unchanged':True,'own_processes_absent':True,'new_forwards':0,'new_environments':0,'server_writes':0}))
