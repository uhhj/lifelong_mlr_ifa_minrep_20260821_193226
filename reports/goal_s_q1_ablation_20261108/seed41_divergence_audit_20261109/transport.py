"""Existing authorized transport, restricted to this CPU read-only audit."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parents[1]/'20261027_support_gate_forks'))
import remote
from credential_store import load
assert sys.argv[1]=='capacity_readonly.py'
remote.R=R
secret=load()
assert secret
remote.getpass.getpass=lambda prompt:secret
try:remote.run(sys.argv[1])
finally:secret=None;remote.getpass.getpass=lambda prompt:None
