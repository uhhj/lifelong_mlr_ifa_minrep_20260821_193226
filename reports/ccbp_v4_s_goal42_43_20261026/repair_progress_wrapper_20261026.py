"""External local wrapper repair only; never modifies frozen experiment sources."""
from pathlib import Path
import json,hashlib,ast,datetime
R=Path(__file__).resolve().parent;D=R/'wrapper_repair_20261026_1456';D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['audit_seed_progress.py','audit_progress.py','prepare_audits.py'];before={}
for name in names:
 p=R/name;before[name]=sha(p);(D/(name+'.before.txt')).write_bytes(p.read_bytes())
p=R/'audit_seed_progress.py';s=p.read_text(encoding='utf8').replace("for j,binding in inp['memory'].items():","for j,memory_binding in inp['memory'].items():").replace("assert binding==dict(cache_sha256=","assert memory_binding==dict(cache_sha256=");p.write_text(s,encoding='utf8')
p=R/'audit_progress.py';s=p.read_text(encoding='utf8').replace("[sys.executable,str(R/", "[sys.executable,'-X','utf8',str(R/")
s=s.replace('capture_output=True,text=True','capture_output=True,text=True,encoding=\'utf8\'');p.write_text(s,encoding='utf8')
p=R/'prepare_audits.py';s=p.read_text(encoding='utf8').replace("put('audit_seed_progress.py',s)","s=s.replace(\"for j,binding in inp['memory'].items():\",\"for j,memory_binding in inp['memory'].items():\").replace(\"assert binding==dict(cache_sha256=\",\"assert memory_binding==dict(cache_sha256=\")\nput('audit_seed_progress.py',s)");p.write_text(s,encoding='utf8')
for name in names:ast.parse((R/name).read_text(encoding='utf8'))
out=dict(status='EXTERNAL_WRAPPER_REPAIR_PENDING_REAL_REAUDIT',time=datetime.datetime.now().astimezone().isoformat(),before=before,after={n:sha(R/n) for n in names},snapshot='check_readonly_20261026_145525',observed_errors=["audit_seed_progress.py line44 KeyError: stages: inner loop binding overwrote parent binding after first nonempty memory.","Parent audit subprocess reader UnicodeDecodeError utf8 byte0xc2: child interpreter lacked explicit UTF8 for Chinese path traceback."],repair='Rename local memory binding and make child interpreter/output UTF8 explicit; scientific source and contract untouched.',server_writes=0,model_calls=0,optimizer_calls=0,environments=0)
(D/'REPAIR.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out,indent=2))
