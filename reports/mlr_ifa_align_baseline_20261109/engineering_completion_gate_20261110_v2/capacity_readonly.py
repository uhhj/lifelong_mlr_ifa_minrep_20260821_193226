from pathlib import Path
import json,hashlib,sys,subprocess,os,time
q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1');p=Path('/data/mlr_ifa/experiments')/q.name
sys.path.insert(0,str(q));from runtime import verify,processes,CASES,sha
verify();assert (q/'controller.exit').read_text().strip()=='1' and not processes([q,p])
artifacts={}
for case in CASES:
 d=p/case
 prohibited=[x for x in d.rglob('*') if x.is_file() and (x.suffix in ['.pth'] or x.name in ['OPTIMIZER_CALLS.json','INITIAL_STATE.json','INITIALIZATION_AUDIT.json','epochs.json'] or x.name.startswith('ADMISSION_'))]
 assert not prohibited,prohibited
 artifacts[case]={str(x.relative_to(d)):sha(x) for x in d.rglob('*') if x.is_file() and x.suffix in ['.log','.exit']}
config=Path('/data/mlr_ifa/config/config_adaptation.yaml');assert config.is_file()
import yaml
cfg=yaml.safe_load(config.read_text());assert all(Path(x).exists() for x in cfg.values())
print(json.dumps(dict(status='PASS_FAILED_IMPORT_BEFORE_INITIALIZATION_ZERO_SCIENCE',observed=time.time(),old_contract=sha(q/'CONTRACT.json'),old_failure=sha(q/'FAILURE.json'),controller_exit=1,processes=[],artifacts=artifacts,optimizer_calls=0,environment_calls=0,config_path=str(config),config_sha256=sha(config),config=cfg)))
