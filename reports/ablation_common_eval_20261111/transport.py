"""Submit reviewed code and collect lightweight evidence through the existing SSH transport."""
import ast
import sys
from pathlib import Path

from common import ROOT, read, sha

REVERSE = ROOT.parent / '20261110_reverse_order_main_protocol'
FILES = ('SPEC.json', 'common.py', 'runtime.py', 'evaluate.py', 'case.py', 'job.py')


def request(operation, authorization=None):
    spec = read(ROOT / 'SPEC.json')
    destination = '/data/mlr_ifa/experiment_workspaces/' + spec['name']
    header = f'from pathlib import Path\nimport json,subprocess,sys\nroot=Path({destination!r})\n'
    if operation == 'prepare':
        files = {name: (ROOT / name).read_bytes() for name in FILES}
        for name, data in files.items():
            assert len(data) < 1024**2
            if name.endswith('.py'):
                ast.parse(data, filename=name, feature_version=(3, 8))
        assert sha(ROOT / 'evaluate.py') == spec['reference_evaluator_sha256']
        contract = dict(files={name: sha(ROOT / name) for name in FILES}, case_count=12, episode_count=2400)
        body = f'''files={files!r}
contract={contract!r}
root.mkdir()
for name,data in files.items():
    (root/name).write_bytes(data)
(root/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\\n')
sys.path.insert(0,str(root))
from common import PYTHON,environment,verify_frozen
verify_frozen()
with (root/'prepare.log').open('x') as log:
    process=subprocess.run([PYTHON,'-B',str(root/'job.py'),'prepare'],cwd=root,env=environment(),stdout=log,stderr=subprocess.STDOUT)
assert process.returncode==0,(root,(root/'prepare.log').read_text(errors='replace')[-2800:])
print((root/'PREPARATION.json').read_text())
'''
    else:
        assert operation in ('launch', 'snapshot', 'final')
        argument = repr(authorization) if operation == 'launch' else ''
        body = f"sys.path.insert(0,str(root))\nfrom job import {operation}\nprint(json.dumps({operation}({argument})))\n"
    # The transport helper accepts a fixed script filename and uses memory-only credentials.
    path = ROOT / ('request_' + operation + '.py')
    path.write_text(header + body, encoding='utf-8')
    sys.path.insert(0, str(ROOT.parent / '20261027_support_gate_forks'))
    import remote
    from credential_store import load

    remote.R = ROOT
    secret = load()
    assert secret
    remote.getpass.getpass = lambda prompt: secret
    try:
        remote.run(path.name)
    finally:
        secret = None
        remote.getpass.getpass = lambda prompt: None


if __name__ == '__main__':
    action = sys.argv[1]
    request(action, read(sys.argv[2]) if action == 'launch' else None)
