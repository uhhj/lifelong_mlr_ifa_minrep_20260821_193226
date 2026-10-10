"""Independent read-only replay of indices and episode aggregation."""
import csv,json,math,random
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime
from .io import write

def exact(r):
    return dict(auc=sum(sum(r[k][j] for k in range(j,4))/(4-j) for j in range(4))/4,
        fwt=sum(r[j][j] for j in range(4))/4,
        nbt=sum(sum(r[j][j]-r[k][j] for k in range(j+1,4))/(3-j) for j in range(3))/3)

def episode_metrics(O,view):
    rows=[json.loads(x) for x in (O/('trials_'+view+'.jsonl')).read_text().splitlines()]
    assert len(rows)==(200 if view=='primary' else 680)
    groups=defaultdict(list)
    for row in rows:
        assert type(row['success']) is bool and row['seed']==42
        groups[row['stage'],row['epoch'],row['task']].append(row)
        assert Path(row['checkpoint']).parent==O/'run_001'
    for key,values in groups.items():
        assert len(values)==20 and sorted(v['initial_state_index'] for v in values)==list(range(20))
    selected={s:99 for s in range(6,10)}
    if view=='secondary':
        selected={s:max((70,75,80,85,90,95,99),key=lambda e:(sum(x['success'] for x in groups[s,e,s]),e)) for s in range(6,10)}
    expected={(s,99,t) for s in range(6,10) for t in range(6,s+1)} if view=='primary' else ({(s,e,s) for s in range(6,10) for e in (70,75,80,85,90,95,99)}|{(s,selected[s],t) for s in range(7,10) for t in range(6,s)})
    assert set(groups)==expected
    matrix=[[None]*4 for _ in range(4)]
    for s in range(6,10):
        for t in range(6,s+1):matrix[s-6][t-6]=sum(x['success'] for x in groups[s,selected[s],t])*5.
    result=dict(exact(matrix),r=matrix,selected_epochs={str(k):v for k,v in selected.items()})
    write(O/('metrics_'+view+'.json'),result)
    attempts=[json.loads(x) for x in (O/('attempts_'+view+'.jsonl')).read_text().splitlines()]
    starts=[x for x in attempts if x['event']=='vector_attempt_start'];ends=[x for x in attempts if x['event']=='vector_attempt_complete']
    key=lambda r:(r['stage'],r['epoch'],r['task'],r['group'])
    assert len(starts)==len(ends)==len(rows)//10
    assert Counter(map(key,starts))==Counter(map(key,ends)) and all(v==1 for v in Counter(map(key,starts)).values())
    assert sum(x['event']=='environment_construct_complete' for x in attempts)==len(rows)//20
    assert sum(x['event']=='environment_closed' for x in attempts)==len(rows)//20
    return result,dict(formal=len(rows),lane_attempts=len(starts)*10,vector_attempts=len(starts),technical_retries=0,uncommitted=0)

def summarize(O,S):
    import torch # CPU-only index verification, never constructs a policy/environment
    totals=Counter();reasons=Counter();movement=[];stage_counts={};score_position=[]
    for stage,batches in [(6,157),(7,227),(8,309),(9,362)]:
        catalog=json.loads((O/('catalog_task%d.json'%stage)).read_text())
        ids=sorted(i for i in catalog if i.startswith('buffer/'))+sorted(i for i in catalog if i.startswith('current/'))
        random.Random(42).shuffle(ids)
        expected_ids=set(ids);step=0;events_count=0
        for epoch in range(100):
            d=json.loads((O/'order_logs'/('task%d'%stage)/('epoch%03d.json'%epoch)).read_text())
            assert d['stage']==stage and d['epoch']==epoch and len(d['batches'])==batches
            gen=torch.Generator();gen.manual_seed(epoch)
            base=torch.randperm(len(ids),generator=gen).tolist()
            assert d['base_indices']==base and d['base_ids']==[ids[i] for i in base]
            assert Counter(d['base_ids'])==Counter(d['actual_ids'])==Counter(expected_ids)
            reconstructed=list(d['base_ids']);events=d['schedules'];events_count+=len(events)
            expected_steps=[k for k in range(epoch*batches,(epoch+1)*batches) if k%500==0] if stage>6 else []
            assert [r['step'] for r in events]==expected_steps
            for r in events:
                start=(r['step']-epoch*batches)*10;end=min(len(ids),start+5000)
                assert r['start']==start and r['end']==end and r['epoch']==epoch
                totals['schedules']+=1
                for t,row in r['tasks'].items():
                    t=int(t);ps=row['positions'];cs=row['candidates'];assert len(ps)==len(set(ps))==len(cs)<=64
                    assert ps==sorted(ps) and all(start<=p<end for p in ps)
                    assert cs==[reconstructed[p] for p in ps]
                    assert all(catalog[i]['task']==t<stage for i in cs)
                    assert all(p%10!=0 and (len(ids)%10==0 or p<len(ids)-len(ids)%10) for p in ps)
                    components=row.get('score_components',[]);totals['candidate_gradients']+=len(components)
                    for c in components:
                        if c['score'] is not None:
                            assert math.isclose(c['score'],c['history_alignment']-c['conflict_penalty'],rel_tol=0,abs_tol=2e-7)
                    if row['committed']:
                        assert len(set(p//10 for p in ps))>1 and len(components)==len(cs)
                        values=[x['score'] for x in components];assert max(values)-min(values)>1e-8
                        assert sorted(row['permutation'])==list(range(len(cs)))
                        ordered=[cs[i] for i in row['permutation']];assert ordered==row['ordered_ids']
                        assert Counter(ordered)==Counter(cs)
                        score_by_id={x['id']:x['score'] for x in components}
                        oldpos={i:p for i,p in zip(cs,ps)}
                        for p,item,move in zip(ps,ordered,row['movements']):
                            advance=oldpos[item]//10-p//10
                            assert move==dict(id=item,from_position=oldpos[item],to_position=p,steps_earlier=advance)
                            reconstructed[p]=item;movement.append(advance);score_position.append([score_by_id[item],advance])
                        totals['committed_tasks']+=1;totals['participating_positions']+=len(ps)
                    else:reasons[row['reason']]+=1
                for t,refs in r.get('references',{}).items():
                    cs=r['tasks'][t]['candidates']
                    assert len(refs)<=16 and len(refs)==len(set(refs)) and set(refs).isdisjoint(cs)
                    assert all(catalog[i]['task']==int(t)<stage for i in refs)
                assert len(r.get('current_reference',[]))<=16
                assert all(catalog[i]['task']==stage for i in r.get('current_reference',[]))
            assert reconstructed==d['actual_ids']==[ids[i] for i in d['actual_indices']]
            actual_stream=[]
            for batch in d['batches']:
                assert batch['step']==step;step+=1
                start=batch['start'];end=batch['end'];assert end-start==len(batch['actual_ids'])<=10
                assert batch['original_ids']==d['base_ids'][start:end] and batch['actual_ids']==d['actual_ids'][start:end]
                assert batch['task_ids']==[catalog[i]['task'] for i in batch['original_ids']]==[catalog[i]['task'] for i in batch['actual_ids']]
                assert batch['original_ids'][0]==batch['actual_ids'][0]
                if len(batch['actual_ids'])<10:assert batch['original_ids']==batch['actual_ids']
                for a,b,t in zip(batch['original_ids'],batch['actual_ids'],batch['task_ids']):
                    if t==stage:assert a==b
                actual_stream+=batch['actual_ids']
            assert actual_stream==reconstructed
            totals['epochs']+=1;totals['actual_slots']+=len(actual_stream)
        assert step==batches*100;stage_counts[str(stage)]=dict(steps=step,schedules=events_count)
        bp=json.loads((O/('buffer_pairing_task%d.json'%stage)).read_text())
        assert bp['members_and_payload_hashes_match'] and bp['count']=={6:781,7:1521,8:2301,9:2961}[stage]
    assert totals['epochs']==400 and totals['schedules']==181 and totals['candidate_gradients']<=24896
    primary,attempts_primary=episode_metrics(O,'primary');secondary,attempts_secondary=episode_metrics(O,'secondary')
    versions={name:json.loads((S/('protective_e1/'+file)).read_text()) for name,file in [('E0','e0_seed42.json'),('E1','e1_seed42.json'),('E1U','e1u_seed42.json')]}
    versions['E1O']=primary;table={k:dict(exact(v['r']),matrix=v['r']) for k,v in versions.items()}
    differences={('E1O_minus_'+k):{metric:table['E1O'][metric]-table[k][metric] for metric in ('auc','fwt','nbt')} for k in ('E0','E1','E1U')}
    train=json.loads((O/'TRAINING_COMPLETE.json').read_text())
    costs=dict(training_invocation_seconds=train['invocation_seconds'],peak_gpu_bytes=train['peak_gpu_bytes'],
        scoring_seconds=sum(v['scoring_seconds'] for v in train['order_totals'].values()),
        reorder_management_seconds=sum(v['management_seconds'] for v in train['order_totals'].values()),
        io_seconds_nested=sum(v['io_seconds_nested'] for v in train['order_totals'].values()),
        checkpoint_and_epoch_log_seconds=sum(json.loads(x)['seconds'] for x in (O/'checkpoint_cost.jsonl').read_text().splitlines()))
    stamps={}
    for line in (O/'commands.log').read_text().splitlines():
        parts=line.split();stamp=datetime.fromisoformat(parts[0]).timestamp()
        if parts[1]=='START':stamps[parts[2]]=stamp
        elif parts[1]=='EXIT' and parts[-1] in ('evaluate_primary','evaluate_secondary'):costs[parts[-1]+'_seconds']=stamp-stamps[parts[-1]]
    stats=dict(cross_batch_moved=sum(v!=0 for v in movement),participating=len(movement),
        mean_absolute_step_movement=sum(abs(v) for v in movement)/max(len(movement),1),
        mean_signed_score_times_advance=sum(s*a for s,a in score_position)/max(len(score_position),1),
        metric_note='Within-permutation earlier/later movement; not evidence of closed-loop efficacy')
    write(O/'order_movement.json',dict(score_and_steps_earlier=score_position,summary=stats))
    result=dict(versions=table,differences=differences,secondary=secondary,coverage=dict(totals),
        stages=stage_counts,skips_and_fallbacks=dict(reasons),movement=stats,cost=costs,
        attempts=dict(primary=attempts_primary,secondary=attempts_secondary),
        primary_auc_exceeds_E0=table['E1O']['auc']>table['E0']['auc'],
        limits=['Single seed; no statistically stable efficacy claim.',
            'Only the fixed limited reordering budget is tested; not all protective sorting methods.',
            'E1/E1U comparisons change candidate source, frequency allocation and order; not same-pool single-variable pairing.',
            'E0 RTX4090, E1/E1U RTX3090; current hardware recorded in contract is RTX4080 SUPER.',
            'E0 historical timing and stage-buffer snapshots unavailable; no reconstruction of author trajectory.'])
    write(O/'COVERAGE_ORDER_AUDIT.json',result)
    (O/'FINAL_REPORT.txt').write_text('\n'.join(['E1-O results; pending independent process/exit acceptance',json.dumps(result,indent=2)]))
    return result
