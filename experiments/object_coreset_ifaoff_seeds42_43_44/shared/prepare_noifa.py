"""Prepare new isolated IFA-off arms. Does not train or modify parent exports."""
import json,hashlib,shutil,difflib,sys
from pathlib import Path
BASE=Path('/data/mlr_ifa/experiments');WORK=Path('/data/mlr_ifa/experiment_workspaces/Coreset_validation_20261014/noifa')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
for seed in (42,43,44):
 old=BASE/f'Coreset_paper_OBJECT_seed{seed}_20261013';oldsrc=Path(f'/data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_seed{seed}_20261013/source')
 out=BASE/f'Coreset_IFAoff_OBJECT_seed{seed}_20261014';src=WORK/f'seed{seed}/source'
 assert not out.exists() and not src.exists(),'Never reconfigure an existing arm'
 oldc=read(old/'CONTRACT.json')
 assert all(sha(oldsrc/n)==h for n,h in oldc['source_sha256'].items())
 shutil.copytree(oldsrc,src,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 out.mkdir(parents=True);(out/'features').symlink_to(old/'features',target_is_directory=True)
 for n in ('FEATURES.json','MODAL_CONTRACT.json','TASKS.json'):shutil.copyfile(old/n,out/n)
 # Preserve the exact exporter modal contract; new training code provenance is a
 # separate contract referring to the unmodified parent feature/modal hashes.
 train=src/'coreset_exp/train.py';before=train.read_text()
 assert before.count('loss=bc+.1*extra;assert torch.isfinite(loss)')==1
 after=before.replace('loss=bc+.1*extra;assert torch.isfinite(loss)','assert self.cfg.ifa_weight==0.0\n        loss=bc;assert torch.isfinite(loss)')
 train.write_text(after)
 common=src/'coreset_exp/common.py';old_common=common.read_text();prefix=old_common[:old_common.index('def verify():')]
 prefix=prefix.replace("assert SEED in (43,44), 'Only explicitly authorized extension seeds'","assert SEED in (42,43,44), 'Authorized no-IFA seeds'")
 if seed==42:prefix=prefix.replace("O=Path(os.environ['CORESET_OUT'])","O=Path(os.environ['CORESET_OUT'])\nSEED=int(os.environ['CORESET_SEED'])\nassert SEED==42\nSELECTOR_SEED=42")
 # Seed42 version may fix SEED=42. It remains42, while43/44 remain env-driven.
 verify='''def verify():
    c=json.loads((O/'CONTRACT.json').read_text())
    assert c['ifa_weight']==0.0 and c['seed']==SEED
    assert source_hashes()==c['source_sha256']
    assert sha(O/'MODAL_CONTRACT.json')==c['modal_sha256']
    parent=Path(c['parent_run'])
    assert sha(parent/'CONTRACT.json')==c['parent_contract_sha256']
    assert sha(O/'FEATURES.json')==sha(parent/'FEATURES.json')==c['feature_manifest_sha256']
    assert sha(parent/'MODAL_CONTRACT.json')==c['modal_sha256']
    m=json.loads((O/'FEATURES.json').read_text())
    assert m['contract_sha256']==c['parent_contract_sha256']
    assert m['modal_sha256']==c['modal_sha256'] and hashes(O/'features')==m['hashes']
    for p,h in c['inputs'].items():assert sha(p)==h,p
    for p,h in c['configs'].items():assert sha(p)==h,p
    return c
'''
 common.write_text(prefix+verify)
 # Disable legacy whole-pipeline/export/eval launchers in this isolated copy.
 for n in ('pipeline.sh','supervise.sh'):(src/'coreset_exp'/n).write_text('#!/usr/bin/env bash\necho "Use the authorized no-IFA queue, not the old driver" >&2\nexit 64\n')
 configs={}
 for arm in ('random','coreset'):
  c=read(old/f'resolved_{arm}.json');c.update(ifa_weight=0.0,run_root=str(out/arm),buffer_dir=str(out/arm/'buffer'),experiment_dir=str(out/arm/'models'),feature_dir=str(out/'features'),contract=str(out/'CONTRACT.json'))
  write(out/f'resolved_{arm}.json',c);configs[str(out/f'resolved_{arm}.json')]=sha(out/f'resolved_{arm}.json')
 assert read(out/'resolved_random.json')['seed']==seed
 source_hashes={str(p.relative_to(src)):sha(p) for p in sorted(src.rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.sh') and '__pycache__' not in p.parts}
 contract={'protocol':'IFA_off_only_v1','seed':seed,'ifa_weight':0.0,'parent_run':str(old),'parent_contract_sha256':sha(old/'CONTRACT.json'),
  'feature_manifest_sha256':sha(old/'FEATURES.json'),'modal_sha256':sha(old/'MODAL_CONTRACT.json'),
  'modal_semantics':'Unmodified historical export contract; training source separately bound here, not relabelled features',
  'source_sha256':source_hashes,'inputs':oldc['inputs'],'configs':configs,'epochs_each':100,'arms':['random','coreset'],'steps_each_arm':105500,
  'expected_same_buffer_members':{arm:{str(t):sha(old/arm/'storage'/f'task{t}.json') for t in range(6,10)} for arm in ('random','coreset')}}
 write(out/'CONTRACT.json',contract);write(out/'RUN_STATUS.json',{'status':'PREPARED_NOT_TRAINED','seed':seed,'gpu_preflight':'PENDING'})
 (src.parent/'train_noifa.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='original/coreset_exp/train.py',tofile='noifa/coreset_exp/train.py')))
 (src.parent/'common_provenance.patch').write_text(''.join(difflib.unified_diff(old_common.splitlines(True),(prefix+verify).splitlines(True),fromfile='original/coreset_exp/common.py',tofile='noifa/coreset_exp/common.py')))
 print('PREPARED',seed,str(out),flush=True)
