"""Authorized memory-only transport; no credentials in files or command args."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'20261027_support_gate_forks'))
import remote
from credential_store import load
assert sys.argv[1] in {'preflight_readonly.py','prepare_remote.py','check_readonly.py','final_readonly.py','deploy_remote.py','cpu_import_probe.py','launch_remote.py','sourced_order_audit_readonly.py','prelaunch_after_preauth_readonly.py','hold_dispatch_protocol_review.py','resume_original_dispatch.py','install_eval_amendment.py'}
remote.R=R
secret=load();assert secret
remote.getpass.getpass=lambda prompt:secret
try:remote.run(sys.argv[1])
finally:secret=None;remote.getpass.getpass=lambda prompt:None
