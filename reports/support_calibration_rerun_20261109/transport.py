"""Use existing authorized Credential Manager transport; no credential logging."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'20261027_support_gate_forks'))
import remote
from credential_store import load
assert sys.argv[1] in {'preflight_readonly.py','check_readonly.py','deploy_remote.py','capacity_readonly.py','final_readonly.py'}
remote.R=R
secret=load()
assert secret,'No stored credential; no remote action'
remote.getpass.getpass=lambda prompt:secret
try:remote.run(sys.argv[1])
finally:secret=None;remote.getpass.getpass=lambda prompt:None
