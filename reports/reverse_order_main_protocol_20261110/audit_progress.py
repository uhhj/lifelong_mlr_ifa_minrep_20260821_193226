"""Independent reverse-order accounting from recorded scalars and immutable prefixes."""
from pathlib import Path
import hashlib,json,math,sys
R=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(e):
    out={}
    for name,x in e['light'].items():
        assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'],name
        out[name]=json.loads(x['text']) if name.endswith('.json') else ([json.loads(s) for s in x['text'].splitlines() if s] if name.endswith('.jsonl') else x['text'].strip())
    return out
def gate_rows(d,e,case):
    rows=[]
    for name,x in sorted(e.get('diagnostic_files',{}).items()):
        if Path(name).name.startswith(case+'_GATES_'):
            p=Path(d)/name;assert sha(p)==x['sha256'];rows.extend(read(p))
    return rows
def audit(directory,previous=None):
    d=Path(directory);e=read(d/'EVIDENCE.json');obj=decode(e);c=read(R/'CONTRACT.json');spec=read(R/'SPEC.json')
    assert e['contract_sha256']==sha(R/'CONTRACT.json')
    assert 'coordinator/FAILURE.json' not in obj,obj.get('coordinator/FAILURE.json')
    total=dict(epochs=0,accepted=0,production_physical=0,admission=0,training_collection=0,formal=0)
    results={}
    for case,entry in spec['cases'].items():
        inp=read(R/'cases'/case/'INPUT.json');method=inp['method'];p=case+'/'
        counts=read(R/'cases'/case/'INPUT_COUNTS.json')
        h=obj.get(p+('native/epochs.json' if method=='MLR' else 'EPOCHS_S.json'),[])
        assert [(r['task' if method=='MLR' else 'stage'],r['epoch']) for r in h]==[(s,ep) for s in range(6,10) for ep in range(100)][:len(h)]
        gs=gate_rows(d,e,case);group={};last=None
        for g in gs:
            st,ep,bi=g['stage'],g['epoch'],g['batch'];nb=inp['batches'][str(st)]
            assert 7<=st<=9 and 0<=ep<100 and bi in range(0,nb,10)
            start=sum(inp['batches'][str(t)]*100 for t in range(6,st))+ep*nb+bi
            assert (g['steps'],g['start_step'],g['end_step'])==(min(10,nb-bi),start,start+min(10,nb-bi))
            key=(st,ep,bi);assert last is None or key>last;last=key
            cc,qq=g['C'],g['Q'];checks=dict(bc_new=qq['bc_new']<=cc['bc_new'],exec_new=qq['exec_new']<=cc['exec_new'],retain_improved=qq['retain_support']<cc['retain_support'])
            assert g['checks']==checks and g['accepted']==('Q' if all(checks.values()) else 'C')
            assert all(math.isfinite(v[k]) for v in [cc,qq] for k in ['bc_new','exec_new','retain_support'])
            group.setdefault((st,ep),[]).append(g)
        acc=phys=blocks=qaccept=0;ad=[];ad_calls=0
        for i,row in enumerate(h):
            st=6+i//100;ep=i%100;nb=inp['batches'][str(st)];acc+=nb;phys+=nb*(2 if method=='DEBR' and st>6 else 1)
            assert row['batches']==nb and row['steps']==acc
            if method=='DEBR':
                assert row['physical_optimizer_calls']==phys
                tot=row['totals'];assert tot['valid']==counts['valid'][str(st)] and tot['invalid']==inp['population'][str(st)]-tot['valid'] and tot['old_eligible']==counts['old_valid'][str(st)]
                g=group.get((st,ep),[]);assert [v['batch'] for v in g]==(list(range(0,nb,10)) if st>6 else [])
                retained=sum(v['steps'] for v in g if v['accepted']=='Q');blocks+=len(g);qaccept+=sum(v['accepted']=='Q' for v in g)
                assert tot['anchor_slots']==4*retained==tot['anchor_actual']+tot['anchor_empty']
                assert row['gate_blocks']==blocks and row['gate_Q_accepts']==qaccept
        for st in range(6,10):
            a=obj.get(p+('ADMISSION_MLR_%d.json'%st if method=='MLR' else 'ADMISSION_S_%d.json'%st))
            if a:
                if method=='MLR':
                    assert a['status']=='PASS_OWN_MAIN_MLR_GPU_ADMISSION' and a['optimizer_calls']==5
                    assert all(a[k] for k in ['direct_first_update_exact','direct_second_update_exact','serialized_next_step_exact','all_state_restored','random_state_restored','actual_population_order_task_and_hash_exact'])
                elif st==6:
                    assert a['status']=='PASS' and a['optimizer_calls']==4
                    assert all(a[k] for k in ['roundtrip_exact','next_step_exact','restore_initial_exact','baseline_or_legacy_update_exact'])
                    short=obj[p+'SHORT_HISTORY_ADMISSION_S.json'];assert short['status']=='PASS' and short['extra_autograd_calls']==8
                else:
                    assert a['status']=='PASS_OWN_GPU_ADMISSION' and a['optimizer_calls']==29 and a['extra_autograd_calls']==11
                    assert all(a[k] for k in ['q_one_original_S_exact','lambda_zero_original_C_exact','ten_step_C_replay_exact','accepted_and_rejected_full_state_restore','measurement_state_exact','probe_all_state_exact'])
                ad.append(st);ad_calls+=a['optimizer_calls']
        assert not h or set(range(6,6+(len(h)+99)//100))<=set(ad)
        if h:
            assert obj[p+'ACTUAL_BUILD_MAPPING.json']['status']=='PASS_ACTUAL_NATIVE_BUILD_RETURNED_REVERSE_BENCHMARK'
        collected=0
        if method=='DEBR':
            for st in range(6,10):
                memory=obj.get(p+'MEMORY_S_%d.json'%st)
                if memory:
                    assert len(memory['rows'])==memory['count']<=128 and memory['training_attempts']==10
                    assert len(set(v['trajectory'] for v in memory['rows']))==memory['collection_successes']
                    outcomes=obj[p+'COLLECTION_OUTCOMES_S_%d.json'%st];assert len(outcomes)==10
                    assert [(v['initial_index'],v['original_task'],v['attempt']) for v in outcomes]==[(i,inp['order'][st],1) for i in range(20,30)]
                    collected+=10
        common=bool(obj.get('coordinator/EVALUATION_PROTOCOL_AMENDMENT.json'))
        outcomes=obj.get(p+('NORMALIZED_OUTCOMES.json' if method=='MLR' else 'OUTCOMES.json'),obj.get(p+'OUTCOMES.json',[]) if common else [])
        if method=='MLR' and not outcomes and not common:
            for st in range(6,10):
                for t in range(6,st+1):
                    outcomes.extend(dict(stage=st,task=t,original_task=inp['order'][t],initial_index=v['initial_state_index'],attempt=1,success=v['success']) for v in obj.get(p+'native/evaluation/trials_stage%d_task%d.jsonl'%(st,t),[]))
        expected=[(s,t,i) for s in range(6,10) for t in range(6,s+1) for i in range(20)]
        assert [(x['stage'],x['task'],x['initial_index']) for x in outcomes]==expected[:len(outcomes)]
        assert all(x['attempt']==1 and x['original_task']==inp['order'][x['task']] and type(x['success']) is bool for x in outcomes)
        if method=='DEBR' or common:
            assert all(1<=x['steps']<=600 and x['termination']==('success' if x['success'] else 'timeout600') and (x['success'] or x['steps']==600) for x in outcomes)
            assert all(len(x['noise_sha256'])==64 and x['env_seed']==inp['seed']+x['initial_index']//10 for x in outcomes)
        if common and method=='MLR' and outcomes:
            declared=obj[p+'EVALUATION_PROTOCOL_DECLARATION.json']
            assert declared['protocol']=='COMMON_DEBR_EVALUATION_20261111_V1'
            assert obj[p+'EVAL_ADMISSION.json']['status']=='PASS'
        for name in [p+'train.exit',p+'evaluate.exit']:
            if name in obj:assert obj[name]=='0',(name,obj[name])
        calls=obj.get(p+'OPTIMIZER_CALLS.json')
        if p+'train.exit' in obj:
            assert len(h)==400 and acc==inp['accepted_budget'] and phys==inp['physical_budget']
            assert calls['status']=='ACTUAL_CALLS_COMPLETED' and calls['calls']==dict(production=phys,admission=20 if method=='MLR' else 91)
        results[case]=dict(epochs=len(h),accepted=acc,production_physical=phys,admission=ad_calls,training_collection=collected,formal=len(outcomes),admitted_stages=ad,gate_blocks_observed=len(gs),gate_Q_accepts_observed=sum(g['accepted']=='Q' for g in gs),train_exit=obj.get(p+'train.exit'),evaluate_exit=obj.get(p+'evaluate.exit'))
        for key in total:total[key]+=results[case][key]
    held=None;closed=0
    for row in obj.get('coordinator/EVENTS.jsonl',[]):
        if row['kind']=='environment_acquired':assert held is None;held=(row['case'],row['purpose'])
        if row['kind']=='environment_released':assert held==(row['case'],row['purpose']);held=None;closed+=1
    if previous:
        old=read(Path(previous)/'EVIDENCE.json');before=decode(old)
        for name,v in before.items():
            if name.endswith(('.jsonl','EPOCHS_S.json','native/epochs.json','OUTCOMES.json')):assert obj[name][:len(v)]==v,('Prefix changed',name)
            elif any(token in name for token in ['ADMISSION_','INITIAL_','STAGE_S_','MEMORY_S_','COLLECTION_OUTCOMES_','TRAIN_S.json','TRAINING_COMPLETE.json']):assert e['light'][name]['sha256']==old['light'][name]['sha256'],('Completed record changed',name)
        for case in spec['cases']:
            oldgs=gate_rows(previous,old,case);assert gate_rows(d,e,case)[:len(oldgs)]==oldgs
    exitcode=obj.get('coordinator/controller.exit')
    if exitcode is not None:assert exitcode=='0',exitcode
    assert e['processes'] or exitcode=='0','Controller missing without natural successful exit'
    result=dict(status='PASS_PROGRESS_NOT_FINAL',observed=e['observed'],contract=e['contract_sha256'],cases=results,totals=total,processes=e['processes'],controller_exit=exitcode,environment_held=held,closed_environment_intervals=closed,prefix_preserved=bool(previous),capacity=e['capacity'])
    (d/'LOCAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':print(json.dumps(audit(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)))
