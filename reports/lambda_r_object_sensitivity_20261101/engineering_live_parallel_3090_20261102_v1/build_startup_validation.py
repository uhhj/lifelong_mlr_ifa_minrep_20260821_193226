"""Compose zero-GPU isolated startup-lock validation using the actual hook source."""
from pathlib import Path
import ast
E=Path(__file__).resolve().parent;hook=(E/'collection_startup_hook.py').read_text(encoding='utf8')
s='''import os,sys,json,time,hashlib,subprocess,site
from pathlib import Path
D=Path('/data/mlr_ifa/diagnostics/live_parallel_lock_20261102_v3');assert not D.exists();D.mkdir()
HOOK=__HOOK__
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pth=Path(site.getsitepackages()[0])/'zzz_mlr_live_parallel_probe_20261102.pth';assert not pth.exists()
targets={}
for case in ['A','B']:
 d=D/case;d.mkdir();targets[str(d/'collect.py')]=case
 (d/'collect.py').write_text("import random,json,time,sys\\nrandom.seed(7642);x=[random.random() for _ in range(8)];time.sleep(.3);print(json.dumps(dict(values=x,path=sys.path)))\\n")
(D/'collection_startup_hook.py').write_text(HOOK)
(D/'HOOK_CONFIG.json').write_text(json.dumps(dict(targets=targets,lock_path=str(D/'collection.lock'))))
body="try:\\n _hook_path="+repr(str(D/'collection_startup_hook.py'))+"\\n exec(compile(open(_hook_path).read(),_hook_path,'exec'),{'__file__':_hook_path,'__name__':'_mlr_startup_hook'})\\nexcept BaseException:\\n import os,traceback\\n traceback.print_exc()\\n os._exit(86)\\n"
line='import sys,os; exec('+repr(body)+') if sys.argv[0] in '+repr(tuple(targets))+' and os.path.isfile('+repr(str(D/'HOOK_ENABLED.json'))+') else None\\n'
baseline=[]
for case in ['A','B']:
 p=subprocess.run([sys.executable,'-B',str(D/case/'collect.py'),'S','8'],cwd=D/case,capture_output=True,text=True);assert p.returncode==0;baseline.append(json.loads(p.stdout))
(D/'CONTRACT.json').write_text(json.dumps(dict(hook_files={str(D/'collection_startup_hook.py'):sha(D/'collection_startup_hook.py'),str(D/'HOOK_CONFIG.json'):sha(D/'HOOK_CONFIG.json'),str(pth):hashlib.sha256(line.encode()).hexdigest()})))
(D/'HOOK_ENABLED.json').write_text('{}');pth.write_text(line);pth_hash=sha(pth)
try:
 children=[subprocess.Popen([sys.executable,'-B',str(D/case/'collect.py'),'S','8'],cwd=D/case,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for case in ['A','B']]
 outputs=[]
 for p in children:
  out,err=p.communicate(timeout=30);assert p.returncode==0,(p.returncode,err);outputs.append(json.loads(out))
 assert outputs==baseline
 rows=[]
 for case in ['A','B']:rows.extend(json.loads(x) for x in (D/(case+'_COLLECTION_LOCK.jsonl')).read_text().splitlines())
 active=None
 for row in sorted(rows,key=lambda x:x['time']):
  if row['event']=='acquired':assert active is None;active=row['case']
  else:assert active==row['case'];active=None
 assert len(rows)==4 and active is None and max(x['wait_seconds'] for x in rows if x['event']=='acquired')>=.2
 # An unmatched new interpreter must not execute the hook or acquire this lock.
 p=subprocess.run([sys.executable,'-B','-c','import sys,json;print(json.dumps(hasattr(sys,"_mlr_collection_lock_handle")))'],capture_output=True,text=True);assert p.returncode==0 and json.loads(p.stdout) is False
 (D/'HOOK_ENABLED.json').unlink();p=subprocess.run([sys.executable,'-B',str(D/'A/collect.py'),'S','8'],cwd=D/'A',capture_output=True,text=True);assert p.returncode==0 and json.loads(p.stdout)==baseline[0] and len((D/'A_COLLECTION_LOCK.jsonl').read_text().splitlines())==2
 result=dict(status='PASS_ZERO_UPDATE_EXACT_STARTUP_LOCK',time=time.time(),hook_sha256=sha(D/'collection_startup_hook.py'),rows=rows,random_values_and_sys_path_exact=True,unmatched_interpreter_unaffected=True,disabled_hook_unaffected=True,optimizer_calls=0,environments=0,production_signals=0,temporary_pth_sha256=pth_hash)
finally:
 assert sha(pth)==pth_hash;pth.unlink()
result['temporary_pth_removed']=not pth.exists();(D/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''.replace('__HOOK__',repr(hook))
ast.parse(s);(E/'validate_startup_lock.py').write_text(s,encoding='utf8')
