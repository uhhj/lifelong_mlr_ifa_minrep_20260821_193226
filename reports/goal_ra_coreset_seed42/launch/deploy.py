"""New isolated workspace installation only, no training."""
from pathlib import Path
import json,hashlib,zipfile,shutil
W=Path('/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018')
B=Path('/root/mlr_ifa/experiment_workspaces/SCv2_Coreset_GOAL_seed42_20261018')
ARCHIVE=Path('/tmp/ra_goal_code_20261018.zip')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert shutil.disk_usage('/root').free>4*1024**3
if W.exists():
    assert not (W/'DEPLOYMENT.json').exists() and not Path('/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018').exists()
    assert set(p.name for p in W.iterdir())<={'source','INSTALL_FAILURE_1.json'},'Unexpected partial workspace'
    assert sha(W/'source/coreset_exp/train.py')==sha(B/'source/coreset_exp/train.py')
    if not (W/'INSTALL_FAILURE_1.json').exists():(W/'INSTALL_FAILURE_1.json').write_text(json.dumps(dict(exit=1,error="AttributeError: PosixPath has no attribute is_relative_to",phase='before archive first file write',formal_updates=0,environment_steps=0)))
else:
    W.mkdir();shutil.copytree(B/'source',W/'source',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','.git','sc_exp'))
(W/'INSTALL_FAILURE_2.json').write_text(json.dumps(dict(exit=1,error='Installer extension allowlist rejected public source LICENSE/txt/sbatch',formal_updates=0,environment_steps=0)))
with zipfile.ZipFile(ARCHIVE) as z:
    manifest=json.loads(z.read('CODE_MANIFEST.json'));assert set(z.namelist())==set(manifest)|{'CODE_MANIFEST.json'}
    for name,h in manifest.items():
        dest=W/name;assert str(dest.resolve()).startswith(str(W.resolve())+'/')
        assert (dest.suffix in {'.py','.sh','.md','.json','.yaml','.txt','.sbatch'} or dest.name=='LICENSE') and '..' not in Path(name).parts
        if dest.exists():
            old=B/name
            assert sha(dest)==h or (old.is_file() and sha(dest)==sha(old)),name
        raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==h
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    (W/'CODE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
proof=json.loads((W/'SOURCE_DIFF.json').read_text())
for n,h in proof['unchanged_core'].items():assert sha(W/'source'/n)==sha(B/'source'/n)==h
for n in ['train.py','evaluate.py']:
    old=(B/'source/coreset_exp'/n).read_text().replace('sc_exp','ra_exp').replace('sc_coreset','ra_coreset').replace('SC_PARENT','RA_PARENT')
    assert old==(W/'source/coreset_exp'/n).read_text()
(W/'DEPLOYMENT.json').write_text(json.dumps(dict(status='INSTALLED_NOT_STARTED',archive_sha256=sha(ARCHIVE),unchanged_core=proof['unchanged_core'],files=manifest),indent=2)+'\n')
print('INSTALLED_NOT_STARTED',W)
