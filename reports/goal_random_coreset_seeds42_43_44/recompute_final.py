"""Independent, standard-library-only audit of downloaded GOAL evidence.

Reads metadata and outcomes; never loads tensors or executes policy/environment code.
Usage: python recompute_final.py [evidence_directory]
Prints JSON to stdout; does not change any input.
"""
import hashlib
import json
import statistics
import sys
from fractions import Fraction as F
from pathlib import Path

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'final_evidence'
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines()]
manifest = read(R/'PACKAGE_MANIFEST.json')
def verify_package():
    for n,v in manifest.items():
        p=R/n
        assert p.stat().st_size == v['bytes'] and sha(p) == v['sha256'], n
verify_package()
q=R/'queue'
assert (q/'queue.exit').read_text().strip() == (q/'wrapper.exit').read_text().strip() == '0'
audit=read(q/'FINAL_QUEUE_ACCEPTANCE.json')
assert audit['status']=='PASS'
assert [audit[k] for k in ('total_epochs','total_optimizer_steps','final_checkpoints','formal_records','vector_attempts')]==[2400,456600,24,1200,120]
aggregate=read(q/'THREE_SEED_SUMMARY.json')
result={'status':'PASS','package_files':len(manifest),'epochs':2400,'optimizer_steps':456600,'final_checkpoints':24,'formal_records':1200,'vector_attempts':120,'technical_exceptions':0,'seeds':{},'stats':{},'task_auc_contributions_C_minus_R':{}}
expected_batches={6:100,7:139,8:185,9:337}
quota={6:499,7:445,8:452,9:985}
paired_states={}
for seed in (42,43,44):
    root=R/f'seed{seed}'
    assert (root/'pipeline.exit').read_text().strip()==(root/'supervisor.exit').read_text().strip()=='0'
    accepted=read(root/'INDEPENDENT_ACCEPTANCE.json')
    assert accepted['status']=='PASS' and accepted['final_checkpoints']==8 and accepted['source_verified']
    assert len(accepted['loaded_downstream_keys'])==8 and set(accepted['loaded_downstream_keys'].values())=={100}
    parity=read(root/'TASK6_TENSOR_PARITY.json')
    assert parity['equal'] and all(v==0 for v in parity['max_abs_diff'].values())
    assert (root/'random/INITIAL_STATE.json').read_bytes()==(root/'coreset/INITIAL_STATE.json').read_bytes()
    contract=read(root/'CONTRACT.json'); modal=read(root/'MODAL_CONTRACT.json')
    assert (contract['seed'],contract['batch'],contract['gamma'],contract['ifa_weight'])==(seed,10,.7,.1)
    assert all(type(v) is bool and v for v in modal['flags'].values())
    assert sha(root/'MODAL_CONTRACT.json')==contract['modal_sha256']
    assert sha(root/'OLD_INPUTS_READONLY.json')==contract['old_readonly_sha256']
    for n,h in contract['source_sha256'].items():
        assert sha(q/f'seed{seed}'/'source'/n)==h,n
    for n,h in contract['configs'].items():
        assert sha(root/Path(n).name)==h,n
    for name in ('resolved_export.json','resolved_random.json','resolved_coreset.json','resolved_eval.json'):
        cfg=read(root/name)
        assert all(cfg['policy'][flag] is True for flag in ('use_vision','use_language','use_state'))
    assert read(root/'GPU_PREFLIGHT.json')['status']=='PASS'
    saved=read(root/'SUMMARY.json'); assert saved['old_inputs_unchanged']
    arms={}; rates={}
    for arm in ('random','coreset'):
        a=root/arm; epochs=read(a/'epochs.json'); complete=read(a/'TRAINING_COMPLETE.json')
        assert len(epochs)==400 and complete['steps']==76100
        cumulative=0
        for i,e in enumerate(epochs):
            task=6+i//100
            assert (e['task'],e['epoch'],e['batches'])==(task,i%100,expected_batches[task])
            cumulative+=expected_batches[task];assert e['steps']==cumulative
        assert cumulative==76100 and len(complete['checkpoints'])==4
        before={}
        for task in range(6,10):
            st=read(a/'storage'/f'task{task}.json');buf=read(a/f'buffer_after_task{task}.json')
            assert st['K']==quota[task] and len(st['selected'])==quota[task]
            assert st['contract_sha256']==sha(root/'CONTRACT.json') and st['modal_sha256']==sha(root/'MODAL_CONTRACT.json')
            selected={x['destination']:x['sha256'] for x in st['selected']}
            assert len(selected)==quota[task] and not set(selected)&set(before)
            before.update(selected)
            assert buf['hashes']==before and buf['total']==len(before)
            assert buf['counts']=={str(t):quota[t] for t in range(6,task+1)}
        r={};environment_seconds=0
        for stage in range(6,10):
            for task in range(6,stage+1):
                p=a/'evaluation'/f'trials_stage{stage}_task{task}.jsonl'; records=rows(p)
                assert len(records)==20 and {x['trial'] for x in records}==set(range(20))
                assert {x['initial_state_index'] for x in records}==set(range(20))
                cp=complete['checkpoints'][f'mlrifa_{stage}_model_ep99.pth']
                for x in records:
                    assert (x['seed'],x['arm'],x['stage'],x['task'],x['epoch'])==(seed,arm,stage,task,99)
                    assert type(x['success']) is bool and 1<=x['vector_group_steps']<=600
                    assert x['checkpoint_sha256']==cp and x['modal_sha256']==sha(root/'MODAL_CONTRACT.json')
                    assert x['loaded_downstream_keys']==100
                    assert x['initial_state_sha256']==x['initial_hashes'][x['initial_state_index']]
                    key=(seed,stage,task,x['trial'])
                    if arm=='random':paired_states[key]=x['initial_state_sha256']
                    else:assert paired_states[key]==x['initial_state_sha256']
                ev=rows(a/'evaluation'/f'attempts_stage{stage}_task{task}.jsonl')
                assert [x['event'] for x in ev]==['environment_construct_start','environment_construct_complete','vector_attempt_start','vector_attempt_complete','vector_attempt_start','vector_attempt_complete','environment_closed']
                environment_seconds+=ev[-1]['unix']-ev[0]['unix']
                for g in (0,1):
                    start,end=ev[2+2*g:4+2*g]
                    assert start['group']==end['group']==g and start['env_seed']==end['env_seed']==seed+g
                    assert start['initial_indices']==list(range(10*g,10*g+10)) and 1<=end['steps']<=600
                    assert len(end['successes'])==10
                    for i,v in enumerate(end['successes']):
                        x=next(x for x in records if x['initial_state_index']==10*g+i)
                        assert v==x['success'] and end['steps']==x['vector_group_steps']
                r[stage,task]=F(sum(x['success'] for x in records),20)
        metrics={'AUC':sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,'FWT':sum(r[t,t] for t in range(6,10))/4,'NBT':sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3}
        for k,v in metrics.items():
            assert F(saved['arms'][arm]['metrics'][k]['fraction'])==v
            assert abs(aggregate['seeds'][str(seed)][arm]['metrics_percent'][k]-float(100*v))<1e-10
        matrix=[[float(100*r[s,t]) if t<=s else None for t in range(6,10)] for s in range(6,10)]
        assert matrix==aggregate['seeds'][str(seed)][arm]['matrix']
        arms[arm]={'metrics':{k:{'fraction':str(v),'percent':float(100*v)} for k,v in metrics.items()},'matrix':matrix,'training_seconds':complete['seconds_this_invocation'],'summed_epoch_seconds':sum(x['seconds'] for x in epochs),'environment_seconds':environment_seconds}
        rates[arm]=r
    arms['C_minus_R']={k:float(100*(F(arms['coreset']['metrics'][k]['fraction'])-F(arms['random']['metrics'][k]['fraction']))) for k in ('AUC','FWT','NBT')}
    result['seeds'][str(seed)]=arms
    result['task_auc_contributions_C_minus_R'][str(seed)]={str(t):float(100*sum(rates['coreset'][s,t]-rates['random'][s,t] for s in range(t,10))/(10-t)/4) for t in range(6,10)}
for group in ('random','coreset','paired_C_minus_R'):
    result['stats'][group]={}
    for k in ('AUC','FWT','NBT'):
        values=[a['C_minus_R'][k] if group=='paired_C_minus_R' else a[group]['metrics'][k]['percent'] for a in result['seeds'].values()]
        stats={'values':values,'mean':statistics.mean(values),'sample_sd_ddof1':statistics.stdev(values)}
        for n in ('mean','sample_sd_ddof1'):assert abs(stats[n]-aggregate['stats'][group][k][n])<1e-10
        result['stats'][group][k]=stats
for source in (audit,aggregate):
    for remote,h in source['inputs'].items():
        tail=remote.split('Coreset_paper_GOAL_seed',1)[1]; seed,rest=tail.split('_20261017/',1)
        assert sha(R/f'seed{seed}'/rest)==h
verify_package()
result['input_hashes_unchanged']=True
result['payload_verification']='Remote frozen summarize verified original/input/feature/buffer payload hashes; local audit verifies downloaded metadata and receipts, without downloading payloads or checkpoints.'
print(json.dumps(result,indent=2))
