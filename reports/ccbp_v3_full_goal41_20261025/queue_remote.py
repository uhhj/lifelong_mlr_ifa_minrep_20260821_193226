"""Queue one reviewed script to an existing local in-memory credential session."""
from pathlib import Path
import hashlib,json,sys,uuid,datetime
R=Path(__file__).resolve().parent;D=R/'.session_control';D.mkdir(exist_ok=True)
name=sys.argv[1]
assert Path(name).name==name and (R/name).is_file()
assert name in {'preflight_readonly.py','check_readonly.py','capacity_readonly.py','deploy_remote.py','final_readonly.py'}
p=D/('request_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.json')
p.write_text(json.dumps(dict(script=name,sha256=hashlib.sha256((R/name).read_bytes()).hexdigest()),indent=2),encoding='utf8')
print(p)
