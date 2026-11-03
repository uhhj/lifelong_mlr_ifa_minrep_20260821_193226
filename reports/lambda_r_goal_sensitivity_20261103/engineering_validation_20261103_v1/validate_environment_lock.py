"""Zero-model, zero-environment subprocess test of the exact GOAL lock wrapper."""
from pathlib import Path
import json,subprocess,sys,hashlib
D=Path(__file__).resolve().parent;T=D/'lock_simulation'
assert not T.exists(),'Single simulation; inspect original result, never overwrite'
T.mkdir();source=(D/'source/environment_locked.py').read_text()
(T/'environment_locked.py').write_text(source)
stub="import os,sys,json,random,time\nrandom.seed(41)\ntime.sleep(.35)\nprint(json.dumps(dict(argv=sys.argv,cwd=os.getcwd(),path=sys.path,values=[random.random() for _ in range(3)]),sort_keys=True))\n"
for case in ['L005_41','L020_41']:
 p=T/case;p.mkdir()
 for name in ['collect','evaluate']:(p/(name+'.py')).write_text(stub)
def plain(case,kind,extra):
 return subprocess.check_output([sys.executable,'-B',str(T/case/(kind+'.py'))]+extra,cwd=T/case,text=True)
expected={('L005_41','collect'):plain('L005_41','collect',['S','6']),('L020_41','evaluate'):plain('L020_41','evaluate',[])}
jobs=[]
for case,kind,extra in [('L005_41','collect',['S','6']),('L020_41','evaluate',[])]:
 p=subprocess.Popen([sys.executable,'-B',str(T/'environment_locked.py'),case,kind]+extra,cwd=T/case,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);jobs.append((case,kind,p))
for case,kind,p in jobs:
 out,err=p.communicate();assert p.returncode==0,(case,err);assert out==expected[case,kind]
events=[json.loads(x) for x in (T/'ENVIRONMENT_LOCK_EVENTS.jsonl').read_text().splitlines()]
assert len(events)==4
assert [x['phase'] for x in events]==['acquired','released','acquired','released']
assert events[1]['time']<=events[2]['time']
for i in [0,2]:assert events[i]['event_id']==events[i+1]['event_id'] and events[i+1]['returncode']==0
result=dict(status='PASS_ZERO_ENV_EXACT_SUBPROCESS_MUTEX',source_sha256=hashlib.sha256(source.encode()).hexdigest(),argv_cwd_sys_path_random_output_exact=True,nonoverlap=True,environments=0,optimizer_calls=0,events=events)
(D/'LOCK_VALIDATION.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
