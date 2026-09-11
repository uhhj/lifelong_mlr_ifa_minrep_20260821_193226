"""Fail-closed validation gate. This script never starts formal training."""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from confirmation_experiment.common import ROOT,RUNS,REPORTS,read,write,sha,environment
from confirmation_experiment.commands import train_command

def execute(cmd,log,env):
    with Path(log).open('w') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def package_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'confirmation_experiment').rglob('*'))
            if p.is_file() and p.suffix in ('.py','.md','.yaml','.json','.ps1') and '__pycache__' not in p.parts}
def run():
    REPORTS.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();env=environment()
    write(REPORTS/'PREFLIGHT_STATUS.json',{'phase':'unit_tests','pid':os.getpid()})
    execute([sys.executable,'-m','unittest','confirmation_experiment.test_protocol','confirmation_experiment.test_core','confirmation_experiment.test_orchestration','-v'],REPORTS/'unit_tests.log',env)
    execute([sys.executable,'-m','unittest','discover','-s','fixed_kappa_experiment/tests','-p','test_*.py','-v'],REPORTS/'legacy_unit_tests.log',env)
    while not (REPORTS/'CACHE_COMPLETE.json').exists():
        if (REPORTS/'PREPARATION_FAILED.json').exists():raise RuntimeError('Cache preparation failed; inspect it first')
        write(REPORTS/'PREFLIGHT_STATUS.json',{'phase':'waiting_for_cache_preparation','pid':os.getpid()})
        time.sleep(10)
    assert read(REPORTS/'CACHE_COMPLETE.json')['status']=='PASS'
    write(REPORTS/'PREFLIGHT_STATUS.json',{'phase':'gpu_parity','pid':os.getpid()})
    for k in (0,1):
        execute([sys.executable,'fixed_kappa_experiment/parity_real.py','--repo',str(ROOT),'--kappa',str(k),
                 '--baseline-config',str(ROOT/'runs/final_paper_text_faithful/mlr/seed42/run_001/config.json'),
                 '--output',str(REPORTS/f'GPU_ENDPOINT_KAPPA{k}.json')],REPORTS/f'gpu_endpoint_{k}.log',env)
    execute([sys.executable,'-m','confirmation_experiment.gpu_parity'],REPORTS/'gpu_parity.log',env)
    assert read(REPORTS/'GPU_PARITY.json')['status']=='PASS'
    write(REPORTS/'PREFLIGHT_STATUS.json',{'phase':'torchrun_smoke','pid':os.getpid()})
    smoke_env=dict(env,CONFIRMATION_EVENT_LOGGING='1',CONFIRMATION_SNAPSHOTS='0',CONFIRMATION_BARRIERS='0')
    for arm in ('mlr','soft'):
        dest=RUNS/'smoke/seed42'/arm
        assert not dest.exists(),'Existing smoke: inspect rather than silently overwrite'
        dest.mkdir(parents=True)
        execute(train_command(42,arm,smoke=True),dest/'train_stdout.log',smoke_env)
    witnesses=[read(RUNS/'smoke/seed42'/a/'TASK6_TRAIN_READY.json') for a in ('mlr','soft')]
    assert witnesses[0]==witnesses[1],('task6 smoke diverged',witnesses)
    soft=RUNS/'smoke/seed42/soft'
    events=[__import__('json').loads(x) for x in (soft/'events.jsonl').read_text().splitlines()]
    assert len(events)==16 and {r['current_task'] for r in events}=={6,7}
    assert all(r['kappa']==.5 for r in events)
    write(REPORTS/'SMOKE.json',{'status':'PASS','seed':42,'steps_per_arm':16,'task6_states_equal':True,
                              'task6_witness':witnesses[0],'soft_event_rows':len(events)})
    frozen=read(REPORTS/'FROZEN_INPUTS.json')
    for name,h in frozen['code_sha256'].items():assert sha(ROOT/name)==h,name
    for seed,hashes in frozen['cache_sha256'].items():
        for name,h in hashes.items():assert sha(ROOT/name)==h,name
    assert sha(ROOT/'confirmation_experiment/PROTOCOL.md')==frozen['protocol_sha256']
    reference_hashes={}
    for name in ('reports/fixed_kappa/fixed_v1/kappa_0p5/results.json',
                 'reports/fixed_kappa/fixed_v1/validation/preflight.json'):
        reference_hashes[name]=sha(ROOT/name)
    for item in frozen['historical_configs'].values():
        assert sha(ROOT/item['path'])==item['sha256']
        reference_hashes[item['path']]=item['sha256']
    result={'status':'PASS','wall_seconds':time.monotonic()-started,'seeds':[45,46,47],
            'arms':['mlr','soft'],'unit_tests':'unit_tests.log','legacy_tests':'legacy_unit_tests.log',
            'gpu_parity_sha256':sha(REPORTS/'GPU_PARITY.json'),'smoke_sha256':sha(REPORTS/'SMOKE.json'),
            'frozen_inputs_sha256':sha(REPORTS/'FROZEN_INPUTS.json'),'package_sha256':package_hashes(),
            'historical_asset_sha256':reference_hashes,
            'formal_task6_gate':'Deferred to formal paired queue; must pass before any task7 update.',
            'formal_training_started':False}
    write(REPORTS/'PREFLIGHT_COMPLETE.json',result)
    write(REPORTS/'PREFLIGHT_STATUS.json',{'phase':'PASS','pid':os.getpid()})
    report='\n\n'.join([
        '**PRECHECK REPORT — PASS**',
        '新seed45/46/47审计通过。只比较MLR与同批聚合κ=0.5。协议、官方checkpoint、原始数据和三seed缓存已固定并完成哈希核对。',
        '单元测试、既有fixed-kappa回归测试、真实GPU κ=0/1端点验证通过。真实策略日志开/关比较的loss、梯度、参数、AdamW、scheduler与RNG逐位一致，含真实触发和机制快照恢复下一步检查。详见GPU_PARITY.json及原始日志。',
        '两arm均通过torchrun真实缓存smoke；task6的逐batch损失序列和model/optimizer/scheduler/RNG哈希完全一致。正式task6的100epoch与评估一致性仍由队列强制闸门检查，不以smoke替代。',
        '三seed缓存的完整CompletePolicy前端配置与历史生成配置逐项相同，use_language=True、use_state=False。BriefPolicy训练端消费固定H，不含上述前端开关。',
        '正式训练尚未由本preflight脚本启动。只有单独正式队列写出STARTED.json并有存活PID/训练日志时，才能报告已启动。',
        '带日志的timing包含诊断复制、范数和快照IO；不是纯算法开销。3个新seed的主报告先于历史及6seed探索性描述；机制数据不得反向影响训练。',
    ])+'\n'
    (REPORTS/'PRECHECK_REPORT.md').write_text(report,encoding='utf-8')

if __name__=='__main__':
    try:run()
    except Exception as e:
        write(REPORTS/'PREFLIGHT_FAILED.json',{'error':repr(e),'pid':os.getpid()})
        (REPORTS/'PRECHECK_REPORT.md').write_text('**PRECHECK REPORT — FAILED; formal queue not permitted**\n\n'+repr(e)+'\n')
        raise
