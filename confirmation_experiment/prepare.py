"""Capture frozen assets and generate each new-seed cache once, before training."""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from confirmation_experiment.common import ROOT,RUNS,REPORTS,read,write,sha,environment
from confirmation_experiment.commands import feature_command
from confirmation_experiment.protocol import SEEDS

CODE_PATHS=['libero/lifelong/algos/fixed_kappa.py','libero/lifelong/algos/mlr_fixed_kappa.py',
            'libero/lifelong/algos/mlr.py','libero/lifelong/algos/base.py','libero/lifelong/algos/__init__.py',
            'libero/lifelong/models/brief.py','libero/lifelong/models/complete_policy.py',
            'libero/lifelong/models/base_policy.py','libero/lifelong/models/policy_head.py',
            'libero/lifelong/lifelong_adaptation.py','libero/lifelong/save_feature.py',
            'libero/lifelong/test.py','libero/lifelong/metric.py','libero/lifelong/utils.py',
            'libero/lifelong/datasets.py']

def run():
    audit=read(REPORTS/'SEED_AUDIT.json')
    assert audit['status']=='PASS' and audit['fixed_seeds']==list(SEEDS)
    assert not (REPORTS/'CACHE_COMPLETE.json').exists(),'Preparation already completed; do not regenerate'
    assert shutil.disk_usage(ROOT).free>60*1024**3,'Need at least 60 GiB before preparing'
    protocol=ROOT/'confirmation_experiment/PROTOCOL.md'
    manifest={'protocol_sha256':sha(protocol),'starting_state':read(REPORTS/'STARTING_STATE.json'),
              'seed_audit_sha256':sha(REPORTS/'SEED_AUDIT.json'),'code_sha256':{},'input_sha256':{},
              'historical_configs':{},'cache_sha256':{},'cache_protocols':{},
              'environment':{'python':sys.version,'platform':platform.platform(),'kernel_nvidia':Path('/proc/driver/nvidia/version').read_text()}}
    for name in CODE_PATHS:manifest['code_sha256'][name]=sha(ROOT/name)
    for p in (ROOT/'libero/configs').rglob('*.yaml'):manifest['code_sha256'][str(p.relative_to(ROOT))]=sha(p)
    ckpt=ROOT/'checkpoints/object_multitask_model_ep99.pth'
    manifest['input_sha256'][str(ckpt.relative_to(ROOT))]=sha(ckpt)
    assert manifest['input_sha256'][str(ckpt.relative_to(ROOT))]=='159351b1df166dd5391c6b1e1317c0ab299fc8b1447fe466acef14ed7896bb7b'
    raw=list((ROOT/'data/libero_object').glob('*.hdf5'))
    assert len(raw)==10,'Expected full ten-task OBJECT demonstration dataset'
    for p in raw:manifest['input_sha256'][str(p.relative_to(ROOT))]=sha(p)
    for directory in ('libero/libero/bddl_files/libero_object','libero/libero/init_files/libero_object'):
        for p in (ROOT/directory).glob('*'):
            if p.is_file():manifest['input_sha256'][str(p.relative_to(ROOT))]=sha(p)
    for seed in (42,43,44):
        for arm,path in [('mlr',ROOT/f'runs/final_paper_text_faithful/mlr/seed{seed}/run_001/config.json'),
                         ('soft',ROOT/f'runs/fixed_kappa/fixed_v1/formal/kappa_0p5/seed{seed}/run_001/config.json')]:
            manifest['historical_configs'][f'{seed}/{arm}']={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
            cfg=read(path)
            # BriefPolicy consumes cached H and has no front-end switches.
            # Validate those switches on the original CompletePolicy cache config.
            assert cfg['policy']['policy_type']=='BriefPolicy'
    feature_reference=ROOT/'runs/object_seed42_feature/run_001/config.json'
    assert feature_reference.is_file(),'Locate historical feature-generation config before proceeding'
    front=read(feature_reference)
    assert front['policy']['use_language'] is True and front['policy']['use_state'] is False
    manifest['feature_reference_config']={'path':str(feature_reference.relative_to(ROOT)),'sha256':sha(feature_reference)}
    manifest['feature_reference_checkpoint_location']={
        'historical_path':front['pretrain'],'current_path':str(ckpt),
        'note':'Historical feature config predates repository relocation; compare checkpoint identity and frozen hash, not the obsolete /data absolute prefix.'}
    assert Path(front['pretrain']).name==ckpt.name
    historical_precheck=ROOT/'reports/fixed_kappa/fixed_v1/validation/preflight.json'
    assert sha(ckpt) in historical_precheck.read_text(),'Official checkpoint must match the previously audited formal protocol'
    manifest['historical_precheck_sha256']=sha(historical_precheck)
    write(REPORTS/'INPUTS_BEFORE_CACHES.json',manifest)
    env=environment()
    import torch
    manifest['environment'].update(torch=torch.__version__,cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0),gpu_bytes=torch.cuda.get_device_properties(0).total_memory)
    gl=read(REPORTS/'GL_RUNTIME.json')
    assert gl['egl_test_code']==0 and gl['matches_apt_metadata'] is True
    manifest['environment']['process_graphics_runtime']=gl
    for seed in SEEDS:
        dest=ROOT/f'features/object_seed{seed}';rundir=RUNS/f'feature_seed{seed}'
        generated=rundir/'CACHE_GENERATED.json'
        cmd=feature_command(seed)
        write(REPORTS/'PREPARATION_STATUS.json',{'phase':'feature_generation','seed':seed,'pid':os.getpid(),'command':cmd})
        started=time.monotonic()
        if dest.exists() or rundir.exists():
            # An already successful generator is audited and reused after a
            # validation failure. Never regenerate it or accept a partial run.
            witness=read(generated)
            assert witness['generator_exit_code']==0 and witness['seed']==seed
            assert witness['config_sha256']==sha(rundir/'run_001/config.json')
            assert witness['stdout_sha256']==sha(rundir/'feature_stdout.log')
        else:
            rundir.mkdir(parents=True)
            with (rundir/'feature_stdout.log').open('w') as log:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            write(generated,{'seed':seed,'generator_exit_code':0,
                'config_sha256':sha(rundir/'run_001/config.json'),'stdout_sha256':sha(rundir/'feature_stdout.log')})
        cfg=read(rundir/'run_001/config.json')
        assert cfg['seed']==seed
        for key in ('policy','data','train','aug','start_task','end_task','training'):
            assert cfg[key]==front[key],('feature protocol drift',seed,key)
        assert Path(cfg['pretrain']).resolve()==ckpt.resolve() and sha(cfg['pretrain'])==sha(ckpt)
        cache={str(p.relative_to(ROOT)):sha(p) for t in (6,7,8,9) for p in sorted((dest/str(t)).glob('*.pt'))}
        counts={str(t):len(list((dest/str(t)).glob('*.pt'))) for t in (6,7,8,9)}
        expected={str(t):len(list((ROOT/f'features/object_seed42/{t}').glob('*.pt'))) for t in (6,7,8,9)}
        assert counts==expected and all(counts.values()),('cache counts',seed,counts,expected)
        manifest['cache_sha256'][str(seed)]=cache
        manifest['cache_protocols'][str(seed)]={'config_sha256':sha(rundir/'run_001/config.json'),
            'counts':counts,'command':cmd,'wall_seconds':time.monotonic()-started,'full_policy_matches_historical_feature_config':True}
        write(REPORTS/f'CACHE_SEED{seed}.json',manifest['cache_protocols'][str(seed)])
    for name,h in manifest['code_sha256'].items():assert sha(ROOT/name)==h
    assert sha(protocol)==manifest['protocol_sha256']
    write(REPORTS/'FROZEN_INPUTS.json',manifest)
    write(REPORTS/'CACHE_COMPLETE.json',{'status':'PASS','seeds':list(SEEDS),'frozen_inputs_sha256':sha(REPORTS/'FROZEN_INPUTS.json')})
    write(REPORTS/'PREPARATION_STATUS.json',{'phase':'COMPLETE','pid':os.getpid(),'seeds':list(SEEDS)})

if __name__=='__main__':
    try:run()
    except Exception as e:
        write(REPORTS/'PREPARATION_FAILED.json',{'error':repr(e),'pid':os.getpid()});raise
