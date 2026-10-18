"""Install reviewed code only into a new dedicated workspace. Never runs training."""
from pathlib import Path
import json,hashlib,zipfile,shutil,subprocess
W=Path('/root/mlr_ifa/experiment_workspaces/SCv2_Coreset_GOAL_seed42_20261018')
B=Path('/root/mlr_ifa/experiment_workspaces/SC_Coreset_GOAL_seed42_20261018')
ARCHIVE=Path('/tmp/scv2_goal_code_20261018.zip')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not W.exists()
assert shutil.disk_usage('/root').free>4*1024**3
W.mkdir();shutil.copytree(B/'source',W/'source',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','.git'))
with zipfile.ZipFile(ARCHIVE) as z:
    manifest=json.loads(z.read('CODE_MANIFEST.json'))
    assert set(z.namelist())==set(manifest)|{'CODE_MANIFEST.json'}
    for name,h in manifest.items():
        dest=W/name;assert str(dest.resolve()).startswith(str(W.resolve())+'/')
        assert dest.suffix in {'.py','.sh','.md','.json','.yaml'} and '..' not in Path(name).parts
        raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==h
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    (W/'CODE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
critical=['coreset_exp/train.py','coreset_exp/evaluate.py','coreset_exp/eval_metric.py','coreset_exp/model.py','coreset_exp/common.py','libero/lifelong/models/complete_policy.py','libero/lifelong/models/brief.py','libero/lifelong/selectors/greedy_kcenter.py']
assert all(sha(W/'source'/n)==sha(B/'source'/n) for n in critical)
evidence=dict(status='INSTALLED_NOT_STARTED',archive_sha256=sha(ARCHIVE),unchanged_core={n:sha(W/'source'/n) for n in critical},files=manifest)
(W/'DEPLOYMENT.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(dict(status='INSTALLED_NOT_STARTED',files=len(manifest),workspace=str(W))))
