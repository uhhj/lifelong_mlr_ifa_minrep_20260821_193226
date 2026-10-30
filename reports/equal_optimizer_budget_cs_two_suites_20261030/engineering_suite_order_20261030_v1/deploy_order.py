"""Deploy only the authorized suite-order coordinator, once, no training restart."""
import os,json,time,hashlib,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/C_MATCH_FULL_S_BUDGET_GOAL_OBJECT_20261030_v1');E=Q/'engineering_suite_order_20261030_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())
def proc(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split()
 return dict(pid=pid,state=s[0],ppid=int(s[1]),start=s[19],exit_code=int(s[49]),cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
assert not E.exists(),'No repeat engineering deployment'
assert sha(Q/'CONTRACT.json')=='f0ab336c42a9e5f36a7e8d659e1c3bec9047fd775db569a68ff04f75c65808b7'
c=js(Q/'CONTRACT.json');assert all(sha(p)==h for p,h in c['files'].items())
s=js(Q/'STATUS.json');assert (s['controller'],s['case'],s['phase'],s['pid'])==(46492,'GOAL_42','train',46718)
old=proc(46492);child=proc(46718);gpu=proc(46723)
assert old['state'] not in ['T','Z'] and old['cmd'].strip().endswith(str(Q/'controller.py'))
assert child['ppid']==46492 and child['state']!='Z' and str(Q/'GOAL_42/train.py') in child['cmd']
assert gpu['ppid']==46718 and gpu['state'] not in ['T','Z']
assert (Q/'GOAL_41/train_C.exit').read_text()=='0' and len(js(Q/'GOAL_41/EPOCHS_C.json'))==700
assert not (Q/'controller.exit').exists() and not (Q/'GOAL_42/train_C.exit').exists()
for case in ['GOAL_41','GOAL_42','GOAL_43','OBJECT_41','OBJECT_42','OBJECT_43']:
 assert not (Q/case/'evaluate.log').exists()
 if case not in ['GOAL_41','GOAL_42']:assert not (Q/case/'train_C.log').exists()
assert set(BUNDLE)=={'controller.py','watchdog.py'}
for name,src in BUNDLE.items():compile(src,name,'exec')
completed={str(p):sha(p) for p in (Q/'GOAL_41').glob('*.json')}
completed.update({str(p):sha(p) for p in (Q/'GOAL_41').glob('*.exit')})
for t in [6,7,8,9]:
 meta=js(Q/'GOAL_41'/('STAGE_C_%d.json'%t));base=Path('/data/mlr_ifa/experiments')/Q.name/'GOAL_41';p=base/('C' if t==9 else 'stage_models/C')/('stage%d.pth'%t)
 assert sha(p)==meta['sha256'];completed[str(p)]=meta['sha256']
E.mkdir();files={}
for n,src in BUNDLE.items():p=E/n;p.write_text(src);files[str(p)]=sha(p)
g=dict(time=time.time(),controller=old,child=child,training=gpu,completed_GOAL41=completed,GOAL42_prefix=js(Q/'GOAL_42/EPOCHS_C.json'),training_signals=0)
(E/'GUARD.json').write_text(json.dumps(g,indent=2));files[str(E/'GUARD.json')]=sha(E/'GUARD.json')
(E/'CONTRACT.json').write_text(json.dumps(dict(files=files,parent_contract=sha(Q/'CONTRACT.json'),scope='Scheduling only: finish GOAL train, then GOAL eval, then OBJECT train/eval',created=time.time()),indent=2))
with (E/'controller.log').open('x') as f:
 p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(E/'controller.py')],cwd=E,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(200):
 if (E/'ARMED.json').exists() or (E/'FAILURE.json').exists():break
 time.sleep(.1)
assert (E/'ARMED.json').exists(),(E/'controller.log').read_text()
assert proc(46492)['state']=='T' and proc(46723)['state'] not in ['T','Z']
print(json.dumps(dict(status='ARMED_WAITING_GOAL42_NATURAL_EXIT',engineering_root=str(E),controller=p.pid,original_controller=46492,training_pid=46723,training_signals=0,engineering_contract=sha(E/'CONTRACT.json'),engineering_files=len(files),armed=js(E/'ARMED.json'),observed=time.time())))
