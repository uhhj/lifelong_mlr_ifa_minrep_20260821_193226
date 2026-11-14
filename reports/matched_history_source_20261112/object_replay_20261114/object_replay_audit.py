"""Audit the replay and compare it with the original 480 OBJECT outcomes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def audit():
    assert sha(ROOT/'CONTRACT.json')==EXPECTED_CONTRACT
    assert sha(ROOT/'object_replay.py')==read(ROOT/'CONTRACT.json')['wrapper_sha256']
    assert (ROOT/'controller.exit').read_text().strip()=='0' and not gpu()
    assert read(ROOT/'STATUS.json')['phase']=='complete'
    for process in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=process.read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
        assert not any(arg.startswith(str(ROOT).encode()+b'/') for arg in args),process
    check_source()
    results={}
    for case in CASES:
        old=SOURCE/'pairs'/case
        new=ROOT/case
        seed=int(case.rsplit('_',1)[1])
        assert (new/'evaluate.exit').read_text().strip()=='0'
        assert sha(new/'INITIAL_SPLIT.json')==sha(old/'INITIAL_SPLIT.json')
        assert read(new/'EVAL_ADMISSION.json')['status']=='PASS'
        ledger=read(new/'INITIAL_SPLIT.json')
        rows=read(new/'OUTCOMES.json')
        assert len(rows)==read(new/'EVALUATION_COMPLETE.json')['episodes']==160

        def key(z):
            return z['arm'],z['task'],z['initial_index']

        past={key(z):z for z in read(old/'OUTCOMES.json')}
        index={key(z):z for z in rows}
        assert len(index)==len(past)==160 and index.keys()==past.keys()
        batches={path.stem:read(path) for path in (new/'outcomes').glob('*.json')}
        assert len(batches)==16 and all(len(batch)==10 for batch in batches.values())
        assert {key(z):z for batch in batches.values() for z in batch}==index
        events=[json.loads(line) for line in (new/'EVAL_EVENTS.jsonl').read_text().splitlines()]
        assert [(z['run'],z['kind']) for z in events]==[
            ('%s_s9_t%d_g%d'%(a,t,g),k) for a in 'ES' for t in (6,7,8,9)
            for g in (0,1) for k in ('start','complete')]
        counts={a:{} for a in 'ES'}
        paired={}
        success_matches=step_matches=action_matches=0
        flips={arm:{task:{'old_success_only':0,'replay_success_only':0} for task in range(6,10)} for arm in 'ES'}
        for task in (6,7,8,9):
            entry=ledger[str(task)]
            path=Path(entry['path'])
            assert sha(path)==entry['file_sha256']
            actual=torch.load(path,map_location='cpu')
            for i in range(20):
                actual_sha=array_sha(np.asarray(actual[i]))
                assert actual_sha==entry['evaluation'][str(i)]
                eps=np.random.RandomState(20261024+seed*1000+task*100+i).standard_normal((600,7)).astype(np.float32)
                for arm in 'ES':
                    z=index[arm,task,i]; p=past[arm,task,i]
                    assert z['noise_sha256']==p['noise_sha256']==array_sha(eps)
                    assert z['initial_sha256']==p['initial_sha256']==actual_sha
                    assert z['initial_file_sha256']==p['initial_file_sha256']==entry['file_sha256']
                    assert z['env_seed']==p['env_seed']==seed+i//10
                    assert z['cp_sha256']==p['cp_sha256']==ACCEPTED[case]['training'][arm]['checkpoint_sha256']
                    assert z['attempt']==1 and z['stage']==9 and 0<z['steps']<=600
                    assert z['termination']==('success' if z['success'] else 'timeout600')
                    assert z['success'] or z['steps']==600
                    success_matches+=z['success']==p['success']
                    step_matches+=z['steps']==p['steps']
                    action_matches+=z['action_stream_sha256']==p['action_stream_sha256']
                    if z['success']!=p['success']:
                        flips[arm][task]['replay_success_only' if z['success'] else 'old_success_only']+=1
            for arm in 'ES':
                counts[arm][task]=sum(index[arm,task,i]['success'] for i in range(20))
            paired[task]={a+'_only':sum(index[a,task,i]['success'] and not index[b,task,i]['success']
                          for i in range(20)) for a,b in ('ES','SE')}
        results[case]=dict(counts=counts,paired=paired,flips=flips,episodes=160,success_matches=success_matches,
            step_matches=step_matches,action_matches=action_matches,outcomes_sha256=sha(new/'OUTCOMES.json'),
            original_outcomes_sha256=sha(old/'OUTCOMES.json'),evaluation_completed_epoch_s=read(new/'EVALUATION_COMPLETE.json')['time'])
    old_diffs=[]; all_diffs=[]
    for v in results.values():
        e,s=(v['counts'][a] for a in 'ES')
        old_diffs.append(sum(s[t]-e[t] for t in (6,7,8))*100/60)
        all_diffs.append(sum(s[t]-e[t] for t in (6,7,8,9))*100/80)
    return dict(status='COMPLETE_ACCEPTED',contract_sha256=EXPECTED_CONTRACT,
        observed_CST=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        formal_episodes=480,controller_exit=0,optimizer_updates=0,results=results,
        oldfinal_paired_pp=dict(mean=mean(old_diffs),sample_sd=stdev(old_diffs)),
        finalall_paired_pp=dict(mean=mean(all_diffs),sample_sd=stdev(all_diffs)))
