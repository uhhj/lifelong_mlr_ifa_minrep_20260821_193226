"""Preserve original diagnostic and make a Python 3.8-compatible external wrapper."""
import ast,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
source=(R/'drift_readonly.py').read_text(encoding='utf8')
assert source.count("k.removeprefix('module.')")==1
source=source.replace("k.removeprefix('module.')", "(k[7:] if k.startswith('module.') else k)")
ast.parse(source)
(R/'drift_readonly_v2.py').write_text(source,encoding='utf8')
(R/'WRAPPER_COMPATIBILITY_V2.json').write_text(json.dumps({
 'status':'ENGINEERING_COMPATIBILITY_ONLY_BEFORE_FIRST_FORWARD',
 'old_script_sha256':hashlib.sha256((R/'drift_readonly.py').read_bytes()).hexdigest(),
 'new_script_sha256':hashlib.sha256((R/'drift_readonly_v2.py').read_bytes()).hexdigest(),
 'change':'Python3.8-compatible optional module. prefix removal; scientific computation unchanged',
 'failed_attempt_forwards':0,'failed_attempt_updates':0,'failed_attempt_environments':0,
 'local_shell_attempt':'Initial local one-liner quoting failed before creating revised file. Canceled pending getpass before credentials or connection; no remote work.'
},indent=2)+'\n',encoding='utf8')
