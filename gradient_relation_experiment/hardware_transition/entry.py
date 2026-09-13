"""Process-local output/cohort adaptation; original frozen mathematical code untouched."""
import argparse
import difflib
import importlib
import sys
import types
import time
from pathlib import Path
from cb_ifa_experiment.gpu_runtime import ROOT,read,write,sha

OLD=ROOT/'reports/gradient_relation_v1'
OUT=ROOT/'reports/gradient_relation_4090_v2'
COHORT='new_controlled_4090_from_official'


def original_integrity():
    f=read(OLD/'FROZEN_EXECUTION.json')
    for rel,h in f['files'].items():assert sha(ROOT/rel)==h,('Original frozen asset changed',rel)


def verify():
    original_integrity()
    f=read(OUT/'FROZEN_EXECUTION.json');assert f['carrier_source']==COHORT
    for rel,h in f['files'].items():assert sha(ROOT/rel)==h,('Transition frozen asset changed',rel)
    assert read(OUT/'PREFLIGHT_COMPLETE.json')['status']=='PASS'


def load(name):
    original_integrity()
    path=ROOT/'gradient_relation_experiment'/f'{name}.py';source=path.read_text(encoding='utf-8')
    adapted=source.replace('runs/gradient_relation_v1','runs/gradient_relation_4090_v2')
    adapted=adapted.replace('new_controlled_3090_from_official',COHORT)
    adapted=adapted.replace('new controlled 3090 carrier','new controlled 4090 carrier')
    if name=='summarize':
        adapted=adapted.replace('当前3090重放历史4090失败已保留，用户随后授权新受控载体。不得将新载体结果写回原历史实验。',
            '硬件中断前3090已完成的24点独立保留；跨硬件续训未通过既有起点核对，因此本组在4090完整执行全部固定120点，不与旧24点混算。')
        adapted=adapted.replace('new controlled3090; not historical4090','new controlled4090; not historical4090')
        adapted=adapted.replace('新生成的3090原生MLR载体','新生成的4090原生MLR载体')
    if name=='test_analysis':adapted=adapted.replace("'synthetic_same_initial'","'synthetic_same_initial'")
    diff=''.join(difflib.unified_diff(source.splitlines(True),adapted.splitlines(True),fromfile=str(path.relative_to(ROOT)),tofile='4090-process/'+name+'.py'))
    (OUT/'adapter_diffs').mkdir(exist_ok=True)
    dp=OUT/'adapter_diffs'/f'{name}.diff'
    if dp.exists():assert dp.read_text()==diff
    else:dp.write_text(diff)
    module=types.ModuleType('gradient_relation_experiment._4090_'+name);module.__file__=str(path)
    exec(compile(adapted,str(path)+'[4090 output adapter]','exec'),module.__dict__)
    module.OUT=OUT
    if hasattr(module,'verify_code'):module.verify_code=verify
    return module


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','control','analysis_test','run','summary']);p.add_argument('--seed',type=int)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.action=='preflight':
        module=importlib.import_module('gradient_relation_experiment.preflight');module.OUT=OUT;module.run()
    elif a.action=='control':
        module=importlib.import_module('gradient_relation_experiment.recover_probe');module.OUT=OUT
        historical=False
        try:module.run(45);historical=True
        except AssertionError as e:
            if 'Historical boundary mismatch' not in str(e):raise
            # Its actual complete native witness is valid as a new-hardware repeat,
            # even when old-hardware equivalence fails. Preserve the failed history claim.
        comparison=read(OUT/'recovery_probe_seed45/WITNESS_COMPARISON.json')
        assert comparison['rebuilt']['batches']==15700 and comparison['rebuilt']['optimizer_step']==15700
        write(OUT/'NATIVE_CONTROL_COMPLETE.json',{'status':'PASS','historical_equivalence':historical,
            'purpose':'complete native4090 task6 witness for independent same-hardware repeat','time':time.time(),
            'witness_sha256':sha(OUT/'recovery_probe_seed45/WITNESS_COMPARISON.json')})
    elif a.action=='analysis_test':
        summary=load('summarize')
        import gradient_relation_experiment
        gradient_relation_experiment.summarize=summary
        sys.modules['gradient_relation_experiment.summarize']=summary
        module=load('test_analysis');module.run()
    elif a.action=='run':
        assert a.seed in (45,46,47);load('run').run(a.seed)
    else:load('summarize').run()


if __name__=='__main__':main()
