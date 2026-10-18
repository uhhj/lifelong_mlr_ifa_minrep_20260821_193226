"""Independently check downloaded progress metadata; never train or evaluate."""
import argparse,hashlib,json,statistics
from pathlib import Path
from datetime import datetime,timezone,timedelta
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--offline',type=Path,required=True);a=parser.parse_args()
    here=Path(__file__).resolve().parent
    for name,v in read(here/'MANIFEST.json').items():
        p=here/name;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],name
    c=read(here/'CHECK.json');epochs=read(here/'epochs.json');trials=read(here/'trials.json');attempts=read(here/'attempts.json')
    assert len(epochs)==400 and sum(e['batches'] for e in epochs)==76100==epochs[-1]['steps']
    assert {(e['task'],e['epoch']) for e in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
    assert c['task6_tensor_equal'] and c['initial_state_equal'] and c['frozen_inputs_pass']
    assert len(c['checkpoint_sha256'])==4 and c['checkpoint_sha256']==c['train_complete']['checkpoints']
    storage=[]
    for row in c['storage']:
        off=read(a.offline/f"task{row['task']}.json")
        assert row['selected_ids']==off['selected_ids']
        assert row['K']==off['q']==len(set(row['selected_ids']))==len(row['buffer_sha256'])
        assert row['swaps']==len(off['swaps'])==len(set(row['selected_ids'])-set(off['original_ids']))
        assert row['before']==off['before'] and row['after']==off['after']
        before,after=row['before'],row['after']
        assert after['unique_times']>=before['unique_times']
        assert set(before['endpoint_demos'])<=set(after['endpoint_demos'])
        assert set(before['demo_counts'])<=set(after['demo_counts'])
        assert after['radius']<=before['radius']+1e-12 and after['mean']<=before['mean']+1e-12
        assert after['E_numerator']<before['E_numerator']
        assert row['ordered_ids_equal'] and row['payload_sha_equal']
        storage.append(dict(task=row['task'],quota=row['K'],changed=row['swaps'],unique=after['unique_times'],offline_sha256=sha(a.offline/f"task{row['task']}.json")))
    assert sum(x['quota'] for x in storage)==c['buffer_files']==2381
    def key(r):return r['stage'],r['task'],r['group']
    starts={key(r):r for r in attempts if r['event']=='vector_attempt_start'}
    complete={key(r):r for r in attempts if r['event']=='vector_attempt_complete'}
    assert len(complete)==c['completed_vectors'] and len(trials)==10*len(complete)==c['committed_records']
    assert len({(r['stage'],r['task'],r['trial']) for r in trials})==len(trials)
    for k,end in complete.items():
        rr=sorted([r for r in trials if (r['stage'],r['task'],r['trial']//10)==k],key=lambda r:r['trial'])
        assert len(rr)==10 and [r['success'] for r in rr]==end['successes']
        duration=next(x for x in c['durations'] if (x['stage'],x['task'],x['group'])==k)
        assert abs(duration['seconds']-(end['unix']-starts[k]['unix']))<1e-6
    cells=[]
    for s in range(6,10):
        for t in range(6,s+1):
            rr=[r for r in trials if (r['stage'],r['task'])==(s,t)]
            if rr:cells.append(dict(stage=s,task=t,n=len(rr),successes=sum(r['success'] for r in rr),complete=len(rr)==20))
    assert cells==c['cells']
    assert c['technical_exceptions']==sum(r['event']=='technical_exception' for r in attempts)==0
    # Task7 vectors are short; unknown remaining task8/9 vectors budgeted as observed full-600 vectors.
    full=statistics.mean(d['seconds'] for d in c['durations'] if d['steps']==600)
    short=statistics.mean(d['seconds'] for d in c['durations'] if d['task']==7)
    remaining={(s,t,g) for s in range(6,10) for t in range(6,s+1) for g in range(2)}-set(complete)
    seconds=sum(short if t==7 else full for s,t,g in remaining)
    for k,r in starts.items():
        if k in remaining:seconds-=min(c['observed_unix']-r['unix'],short if k[1]==7 else full)
    tz=timezone(timedelta(hours=8))
    result=dict(status='PASS_PROGRESS_AUDIT_NOT_FINAL_ACCEPTANCE',observed_cst=datetime.fromtimestamp(c['observed_unix'],tz).isoformat(),
        epochs=400,steps=76100,committed_records=len(trials),completed_vectors=len(complete),storage=storage,
        actual_buffer_payload_hash_check='Server read-only check; local hash metadata and ordered-ID comparison independently verified',
        full_vector_seconds=full,task7_vector_seconds=short,estimated_remaining_evaluation_minutes=seconds/60,
        estimated_eval_end_cst=datetime.fromtimestamp(c['observed_unix']+seconds,tz).isoformat(),
        final_acceptance_allowance_minutes=[3,10],reported_completion_window_cst=['2026-10-18 22:10','2026-10-18 22:30'],
        cells=cells,ifa_selected_pairs_equal=[x['current']['info']['selected_pairs']==x['parent']['info']['selected_pairs'] for x in c['ifa']])
    (here/'LOCAL_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
