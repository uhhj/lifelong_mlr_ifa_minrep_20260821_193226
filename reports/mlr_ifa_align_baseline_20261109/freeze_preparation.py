"""Freeze reviewed prepared text without deploying or allocating a GPU."""
from pathlib import Path
import ast,json,hashlib,datetime
R=Path(__file__).resolve().parent;A=json.loads((R/'ACTIVE_ROOT.json').read_text(encoding='utf8'))
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
bundle={};code={};private={}
for name in ['deploy_remote.py','runtime.py','controller.py','evaluate_locked.py','observe.py','final_case_cpu.py','final_driver.py']:
 text=(R/name).read_text(encoding='utf-8-sig');ast.parse(text,feature_version=(3,8));bundle[name]=text;code[name]=sha(text)
for p in sorted((R/'cases').rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts:continue
 rel=p.relative_to(R/'cases').as_posix();text=p.read_text(encoding='utf-8-sig')
 if '/source/' in rel:
  name='cases/'+rel;code[name]=sha(text)
 else:name='private/'+rel;private[rel]=sha(text)
 if p.suffix=='.py':ast.parse(text,feature_version=(3,8))
 if p.suffix=='.json':json.loads(text)
 bundle[name]=text
contract=dict(protocol='MLR_IFA_RANDOM_PLUS_ALIGN_ONLY_v1',code_files=code,private_files=private,cases=list(json.loads((R/'INPUT_SPEC.json').read_text(encoding='utf8'))),epochs=2400,production_optimizer=544800,admission_optimizer=120,training_environments=0,formal_episodes=1200)
bundle['CONTRACT.json']=json.dumps(contract,indent=2)+'\n';contract_sha=sha(bundle['CONTRACT.json'])
assert 'POST_Q1_ACCEPTANCE.json' not in bundle
(R/'DEPLOY_TEXT_BUNDLE.json').write_text(json.dumps(bundle),encoding='utf8')
(R/'PREPARED_CONTRACT.json').write_text(bundle['CONTRACT.json'],encoding='utf8')
# Small read-only launchers carry the expected contract before importing remote helpers.
header="from pathlib import Path\nimport sys,json,hashlib\nQ=Path("+repr(A['remote_root'])+")\nassert hashlib.sha256((Q/'CONTRACT.json').read_bytes()).hexdigest()=="+repr(contract_sha)+"\nsys.path.insert(0,str(Q))\n"
(R/'check_readonly.py').write_text(header+"from observe import snapshot\nprint(json.dumps(snapshot()))\n",encoding='utf8')
(R/'final_readonly.py').write_text(header+"import subprocess\nfrom runtime import PY,processes,P\nassert (Q/'controller.exit').read_text().strip()=='0' and not processes([Q,P])\np=subprocess.run([PY,'-B',str(Q/'final_driver.py')],cwd=Q,capture_output=True,text=True)\nassert p.returncode==0,p.stderr\nr=json.loads(p.stdout.strip().splitlines()[-1])\nfrom observe import snapshot\ns=snapshot();s.update(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',cpu=r['results'],remote_cpu_directory=r['directory'])\nprint(json.dumps(s))\n",encoding='utf8')
review=dict(status='PREPARED_STATIC_ONLY_NOT_DEPLOYED',observed=datetime.datetime.now().astimezone().isoformat(),contract_sha256=contract_sha,files=len(bundle),bytes=sum(len(x.encode()) for x in bundle.values()),files_sha256={n:dict(bytes=len(s.encode()),sha256=sha(s)) for n,s in bundle.items()},new_gpu_calls=0,new_optimizer=0,new_environments=0)
(R/'UPLOAD_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf8')
for p in R.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
print(json.dumps({k:v for k,v in review.items() if k!='files_sha256'}))
