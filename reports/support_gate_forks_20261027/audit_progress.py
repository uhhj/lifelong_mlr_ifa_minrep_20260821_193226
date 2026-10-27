"""Independent lightweight accounting; never declares scientific acceptance."""
from pathlib import Path
import json,hashlib,sys,math,datetime
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def js(n):return json.loads(v['light'][n]['text'])
for n,x in v['light'].items():assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'],n
cases=json.loads((R/'CASES.json').read_text(encoding='utf8'));counts={};admissions=0;formal=0;accepted=physical=epochs=0;failures={}
for name,c in cases.items():
    b=json.loads((R/name/'BINDING.json').read_text(encoding='utf8'));pop=b['pop'][str(c['stage'])];batch=(pop+9)//10;sub={};common=set()
    for arm in c['arms']:
        n=name+'/EPOCHS_'+arm+'.json';h=js(n) if n in v['light'] else []
        for i,row in enumerate(h):
            assert row['epoch']==i and row['stage']==c['stage'] and row['batches']==batch
            assert row['steps']==b['checkpoint_steps']+(i+1)*batch
            assert row['physical_optimizer_calls']==(i+1)*batch*(2 if arm=='G' else 1)
            assert row['totals']['valid']==b['valid'][str(c['stage'])] and row['totals']['old_eligible']==b['old'][str(c['stage'])]
            assert row['totals']['invalid']==pop-b['valid'][str(c['stage'])]
        sub[arm]=dict(epochs=len(h),accepted_updates=len(h)*batch,physical_optimizer_calls=len(h)*batch*(2 if arm=='G' else 1),last_epoch_seconds=h[-1]['seconds'] if h else None)
        accepted+=sub[arm]['accepted_updates'];physical+=sub[arm]['physical_optimizer_calls'];epochs+=len(h)
        n=name+'/ADMISSION_'+arm+'.json'
        if n in v['light']:
            a=js(n);assert a['status']=='PASS_OWN_GPU_ADMISSION' and a['optimizer_calls']==29 and a['extra_autograd_calls']==11;admissions+=29
            assert a['q_one_original_S_exact'] and a['lambda_zero_original_C_exact'] and a['ten_step_C_replay_exact'] and a['measurement_state_exact']
        n=name+'/INITIAL_'+arm+'.json'
        if n in v['light']:common.add(js(n)['common_state_sha256'])
    assert len(common)<=1
    n=name+'/GATES.jsonl'
    if n in v['light']:
        gates=[json.loads(s) for s in v['light'][n]['text'].splitlines()];previous=b['checkpoint_steps']
        for g in gates:
            assert g['start_step']==previous and 1<=g['steps']<=10 and g['end_step']==previous+g['steps'];previous=g['end_step']
            cc,qq=g['C'],g['Q'];checks=dict(bc_new=qq['bc_new']<=cc['bc_new'],exec_new=qq['exec_new']<=cc['exec_new'],retain_improved=qq['retain_support']<cc['retain_support'])
            assert checks==g['checks'] and g['accepted']==('Q' if all(checks.values()) else 'C')
            assert all(math.isfinite(x[k]) for x in [cc,qq] for k in ['bc_new','exec_new','retain_support'])
        sub['gate_blocks_observed']=len(gates);sub['gate_Q_accepts']=sum(g['accepted']=='Q' for g in gates)
    n=name+'/OUTCOMES.json'
    if n in v['light']:
        records=js(n);seen=set()
        for x in records:
            key=(x['arm'],x['task'],x['initial_index']);assert key not in seen;seen.add(key)
            assert x['arm'] in c['arms'] and x['stage']==c['stage'] and 6<=x['task']<=c['stage'] and 0<=x['initial_index']<20 and x['attempt']==1
            assert 1<=x['steps']<=600 and x['termination']==('success' if x['success'] else 'timeout600')
            assert x['success'] or x['steps']==600
        sub['formal_episodes']=len(records);formal+=len(records)
    counts[name]=sub
for n,x in v['light'].items():
    if n.endswith('.exit') and x['text'].strip()!='0':failures[n]=x['text']
out=dict(status='ENGINEERING_FAILURE_REQUIRES_REVIEW' if failures else 'PASS_PROGRESS_NOT_FINAL',observation=datetime.datetime.fromtimestamp(v['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),counts=counts,epochs=epochs,accepted_updates=accepted,physical_training_optimizer_calls=physical,own_isolated_optimizer_calls=admissions,formal_episodes=formal,failures=failures,processes=v['processes'],capacity=v.get('capacity',{}),snapshot_sha256=sha(D/'EVIDENCE.json'),contract=v.get('contract',v.get('contract_sha256')))
(D/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out,indent=2))
