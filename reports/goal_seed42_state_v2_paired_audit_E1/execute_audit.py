"""Run the supplied read-only package with recorded provenance and exit codes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[1]/'publish_mlr_branch'
ROOT=REPO/'reports/goal_seed42_state_v2/run'
TOOLS=BASE/'tools/MLR_E1_Paired_Audit'
OUT=BASE/'results'
FIXED='eb15698ae8ee356f6098b9777e5c7dad21450b0e'
def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha(path): return sha_bytes(path.read_bytes())
def write(path,data):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO)
def snapshot():
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file())
    assert not any(p.suffix in {'.pth','.pt','.h5','.hdf5'} for p in paths)
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

try:
    package=json.loads((TOOLS/'PACKAGE_SHA256.json').read_text())
    assert all(sha(TOOLS/name)==digest for name,digest in package.items()),'Package hash mismatch'
    head=git('rev-parse','HEAD').decode().strip()
    assert head==FIXED,'Publication HEAD mismatch'
    before=snapshot()
    inputs=['evaluation/rollouts.jsonl','metrics.json','final_acceptance.json','checkpoints/manifest.json',
            'modal_contract.json','source.patch','evaluation/success_matrix.csv','resolved_eval.yaml',
            'data_manifest.json','pretrained_manifest.json','logs/commands.log']
    git_verified={}
    for name in inputs:
        rel=(ROOT/name).relative_to(REPO).as_posix()
        digest=sha_bytes(git('show',FIXED+':'+rel))
        assert digest==before[name], 'Working evidence differs from fixed Git blob: '+name
        git_verified[name]=digest
    provenance={'status':'VERIFIED','checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'actual_repository':str(REPO),'actual_evidence_root':str(ROOT),
        'actual_git_head':head,'actual_git_branch':git('branch','--show-current').decode().strip(),
        'remote_origin':git('remote','get-url','origin').decode().strip(),
        'git_status_before':git('status','--porcelain').decode(),
        'upstream_experiment_base':'a721807f2f7fa7f9833dcc85c592d78e9accd00b',
        'input_files_equal_fixed_commit_git_blobs':git_verified,
        'all_lightweight_run_files_sha256_before':before,
        'package_zip_sha256':sha(Path('C:/Users/Lenovo/Downloads/MLR_E1_Paired_Audit.zip')),
        'package_members_sha256_verified':package,
        'python_executable':sys.executable,'python_version':sys.version,
        'scope':'Existing lightweight evidence only. No checkpoint loading, network, training, or environment execution.'}
    write(BASE/'source_provenance.json',provenance)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    jobs=[('software_tests',[sys.executable,'-B','-m','unittest','discover','-s',str(TOOLS),'-p','test_*.py','-v']),
          ('real_paired_analysis',[sys.executable,'-B',str(TOOLS/'analyze_paired_closed_loop.py'),
                                   '--evidence-root',str(ROOT),'--out-dir',str(OUT)])]
    journal=[]
    for name,cmd in jobs:
        start=datetime.now(timezone.utc).isoformat()
        with (BASE/'logs'/f'{name}.log').open('xb') as log:
            proc=subprocess.run(cmd,cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT)
        row={'name':name,'argv':cmd,'windows_command_line':subprocess.list2cmdline(cmd),
             'cwd':str(BASE),'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),
             'exit_code':proc.returncode,'log':str(BASE/'logs'/f'{name}.log')}
        journal.append(row)
        with (BASE/'execution_commands.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(name,'EXIT',proc.returncode,flush=True)
        assert proc.returncode==0,name+' failed; do not repair input'
    after=snapshot()
    assert before==after,'Input evidence changed'
    assert all(sha(TOOLS/name)==digest for name,digest in package.items()),'Package modified'
    assert git('rev-parse','HEAD').decode().strip()==FIXED
    acceptance=json.loads((OUT/'audit_acceptance.json').read_text(encoding='utf-8'))
    assert acceptance['status']=='PASS' and acceptance['input_hashes_before']==acceptance['input_hashes_after']
    for name,digest in acceptance['output_hashes'].items(): assert sha(OUT/name)==digest
    write(BASE/'execution_verification.json',{'status':'PASS','all_input_run_files_unchanged':True,
        'verified_run_file_count':len(after),'all_lightweight_run_files_sha256_after':after,
        'actual_git_head_after':FIXED,'git_status_after':git('status','--porcelain').decode(),
        'package_unmodified':True,'supplied_tests_exit_code':0,'real_analysis_exit_code':0,
        'new_training_steps':0,'new_rollouts':0,'checkpoint_binaries_loaded':0,
        'analysis_output_hashes_verified':True})
    print('FINAL_READ_ONLY_VERIFICATION_PASS',len(after),'input files',flush=True)
except Exception as exc:
    with (BASE/'BLOCKED.md').open('x',encoding='utf-8') as f:
        f.write('# BLOCKED\n\n'+type(exc).__name__+': '+str(exc)+'\n\nNo input repairs were performed.\n')
    raise
