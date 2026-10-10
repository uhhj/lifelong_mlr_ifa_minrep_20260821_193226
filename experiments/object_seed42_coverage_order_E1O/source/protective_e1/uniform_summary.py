"""Full-log read-only validation and unrounded three-version comparison."""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from .io import write

def exact(r):
    return dict(auc=sum(sum(r[k][j] for k in range(j,4))/(4-j) for j in range(4))/4,
        fwt=sum(r[j][j] for j in range(4))/4,
        nbt=sum(sum(r[j][j]-r[k][j] for k in range(j+1,4))/(3-j) for j in range(3))/3)

def summarize(O,S):
    refs={name:json.loads(path.read_text()) for name,path in [
        ('E0',S/'protective_e1/e0_seed42.json'),('E1',S/'protective_e1/e1_seed42.json'),('E1U',O/'metrics_primary.json')]}
    table={name:dict(exact(item['r']),matrix=item['r']) for name,item in refs.items()}
    differences={label:{k:table[a][k]-table[b][k] for k in ('auc','fwt','nbt')}
        for label,a,b in [('E1_minus_E1U','E1','E1U'),('E1U_minus_E0','E1U','E0')]}
    flows=defaultdict(Counter);gates=Counter();refreshes=Counter();steps=Counter();pool_counts=defaultdict(Counter)
    active={};pool_info={};pairing=[];fallbacks=[];slot_counts=Counter()
    for stage,batches in [(6,157),(7,227),(8,309),(9,362)]:
        catalog=json.loads((O/('catalog_task%d.json'%stage)).read_text())
        for epoch in range(100):
            pp=json.loads((O/'pairing'/('task%d_epoch%03d.json'%(stage,epoch))).read_text());pairing.append(pp)
            p=O/'selection_logs'/('task%d'%stage)/('epoch%03d.jsonl'%epoch)
            rows=[json.loads(x) for x in p.read_text().splitlines()];assert len(rows)==batches
            for event in rows:
                steps[stage]+=1;assert event['stage_optimizer_step']==steps[stage]
                r=event['refresh']
                if r is not None:
                    refreshes[stage]+=1;assert r['step']==event['stage_optimizer_step']-1
                    assert r['step']%500==0
                    if r['reason']!='OK' or any(r['invalid_candidates'].values()):fallbacks.append(dict(stage=stage,step=r['step'],record=r))
                    for task,ids in r['valid_ids'].items():
                        ps=r['probabilities'][task];assert ps==[1./len(ids)]*len(ids)
                        key=(stage,r['step'],int(task));active[stage,int(task)]=key
                        pool_info[key]=dict(ids=ids,scores=r['scores'][task],n=len(ids),ess=1/sum(p*p for p in ps))
                samples=event['actual_samples'];assert samples is not None and len(samples)<=10
                replacements={e['slot']:e for e in event['replacements']}
                for slot,row in enumerate(samples):
                    task=row['task'];assert row['slot']==slot
                    assert catalog[row['original_id']]['task']==task==catalog[row['final_id']]['task']
                    assert row['attempted']==(slot in replacements)
                    assert row['id_changed']==(row['original_id']!=row['final_id'])
                    if task==stage:assert not row['attempted'] and not row['id_changed']
                    else:
                        slot_counts[stage,task]+=1
                        flows[stage,task][row['final_id']]+=1
                    e=replacements.get(slot)
                    if e:
                        gates['attempted']+=1;gates['id_changed']+=row['id_changed'];gates['ifa_veto']+=row['ifa_veto']
                        assert row['final_id']==(e['candidate'] if e['applied'] else row['original_id'])
                        key=active[stage,task];assert e['candidate'] in pool_info[key]['ids']
                        pool_counts[key][e['candidate']]+=1
        assert steps[stage]==100*batches
        bp=json.loads((O/('buffer_pairing_task%d.json'%stage)).read_text());assert bp['members_and_payload_hashes_match']
        assert bp['count']=={6:781,7:1521,8:2301,9:2961}[stage]
    assert dict(refreshes)=={7:46,8:62,9:73};assert sum(steps.values())==105500
    pools=[];top_count=0;draws=0
    for key,info in pool_info.items():
        counts=pool_counts[key];n=sum(counts.values());size=info['n']
        top=set(sorted(info['ids'],key=lambda k:info['scores'][info['ids'].index(k)],reverse=True)[:size//4])
        top_count+=sum(v for k,v in counts.items() if k in top);draws+=n
        pools.append(dict(stage=key[0],refresh_step=key[1],task=key[2],candidate_count=size,ess=info['ess'],draws=n,
            unique=len(counts),actual_repeat_fraction=(n-len(counts))/n if n else None,
            expected_uniform_unique=size*(1-(1-1/size)**n)))
    flow_summary=[dict(stage=k[0],task=k[1],slots=sum(c.values()),unique=len(c),repeat_fraction=1-len(c)/sum(c.values()),
                       most_common=c.most_common(10)) for k,c in sorted(flows.items())]
    write(O/'pool_diagnostics.json',pools);write(O/'full_replay_flow.json',flow_summary)
    attempts={}
    for view,total in [('primary',200),('secondary',680)]:
        rows=[json.loads(x) for x in (O/('attempts_'+view+'.jsonl')).read_text().splitlines()]
        starts=[r for r in rows if r['event']=='vector_attempt_start'];ends=[r for r in rows if r['event']=='vector_attempt_complete']
        key=lambda r:(r['stage'],r['epoch'],r['task'],r['group'])
        assert len(starts)==len(ends)==total//10 and Counter(map(key,starts))==Counter(map(key,ends))
        assert all(v==1 for v in Counter(map(key,starts)).values())
        assert sum(r['event']=='environment_construct_complete' for r in rows)==total//20
        assert sum(r['event']=='environment_closed' for r in rows)==total//20
        formal=[json.loads(x) for x in (O/('trials_'+view+'.jsonl')).read_text().splitlines()]
        assert len(formal)==total
        attempts[view]=dict(vector_attempts=len(starts),lane_attempts=len(starts)*10,formal_records=total,uncommitted_attempts=0)
    train=json.loads((O/'TRAINING_COMPLETE.json').read_text())
    cost=dict(training_seconds=train['invocation_seconds'],peak_gpu_bytes=train['peak_gpu_bytes'],
       scoring_seconds=sum(v['scoring_seconds'] for v in train['selection_totals'].values()),
       replacement_seconds=sum(v['replacement_seconds'] for v in train['selection_totals'].values()),
       io_seconds_nested_in_scoring_and_replacement=sum(v['io_seconds'] for v in train['selection_totals'].values()),
       checkpoint_and_epoch_log_seconds=sum(json.loads(x)['seconds'] for x in (O/'checkpoint_cost.jsonl').read_text().splitlines()),
       note='Component times are included in training wall time; do not add IO again. Interrupted attempts retained separately; E0 historic wall time unavailable.')
    stamps={}
    for line in (O/'commands.log').read_text().splitlines():
        parts=line.split();stamp=datetime.fromisoformat(parts[0]).timestamp()
        if parts[1]=='START':stamps[parts[2]]=stamp
        elif parts[1]=='EXIT' and parts[-1] in ('evaluate_primary','evaluate_secondary'):cost[parts[-1]+'_seconds']=stamp-stamps[parts[-1]]
    result=dict(versions=table,differences=differences,primary_auc_exceeds_E0=table['E1U']['auc']>table['E0']['auc'],
        strict_pairing=all(x['strict_pairing'] for x in pairing),pairing_epochs=len(pairing),
        pairing_mismatches=[x for x in pairing if not x['strict_pairing']],numeric_fallbacks=fallbacks,
        refreshes=dict(refreshes),steps=dict(steps),gates=dict(gates),priority_draws=draws,
        top_quartile_observed_share=top_count/draws,top_quartile_expected_share=.25,
        attempts=attempts,cost=cost,
        limits=['Single seed observation, not a statistically stable or causal effect estimate.',
          'E1U-E0 combines pool restriction and replacement structure; not a pure pool-size effect.',
          'E1 lacks retained-slot IDs; no invented full-flow E1 coverage comparison.',
          'Scoring is retained as a sidepass, so this is not a scoring-free efficiency version.',
          'E0 RTX4090 versus E1/E1U RTX3090; E0 timing and historical stage buffer snapshots unavailable.'])
    write(O/'UNIFORM_AUDIT.json',result)
    lines=['E1-U results (pending external process/exit acceptance)','',json.dumps(table,indent=2),json.dumps(differences,indent=2),json.dumps(cost,indent=2),'',*result['limits']]
    (O/'FINAL_REPORT.txt').write_text('\n'.join(lines))
    return result
