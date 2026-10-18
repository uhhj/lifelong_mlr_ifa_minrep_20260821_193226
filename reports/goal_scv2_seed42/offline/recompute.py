"""Independent metadata-only acceptance. Does not import the selector.

Replays identities, integer E, support and quotas locally. Numeric geometry
witnesses come from remote direct full-distance recomputation, not new inference.
"""
import json,hashlib,csv
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
E=ROOT/'evidence'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))

def check(r):
    meta=r['metadata'];ids=sorted(meta);q=r['q']
    assert len(ids)==r['N'] and q==(r['N']+1)//2
    def uid(s):return meta[s]['demo_id'],meta[s]['target']
    def group(s):
        a=meta[s];b=min(3,4*a['target']//max(1,a['length']-1)) if a['length']>1 else 3
        return b*3+a['group']%3
    unique={}
    for s in ids:
        if uid(s) in unique:assert unique[uid(s)]==group(s)
        unique[uid(s)]=group(s)
    ref=Counter(unique.values());U=len(unique)
    assert U==r['reference_unique_total']
    assert {int(k):v for k,v in r['reference_unique_counts'].items()}==ref
    def stats(seq):
        counts=Counter(group(s) for s in seq)
        err=sum(abs(counts[g]*U-q*n) for g,n in ref.items())
        return dict(E_numerator=err,E_denominator=2*q*U,E=err/(2*q*U),
            unique_times=len({uid(s) for s in seq}),group_counts={str(k):v for k,v in counts.items()},
            demo_counts=dict(Counter(meta[s]['demo_id'] for s in seq)),
            endpoint_demos=sorted({meta[s]['demo_id'] for s in seq if meta[s]['target']==meta[s]['length']-1}),
            endpoint_windows=sum(meta[s]['target']==meta[s]['length']-1 for s in seq))
    seq=r['original_ids'].copy();assert len(seq)==len(set(seq))==q
    initial=stats(seq)
    for k,v in initial.items():assert v==r['before'][k],k
    old=initial
    for step,w in enumerate(r['swaps']):
        x,y=w['added'],w['removed'];assert w['step']==step and seq[w['slot']]==y
        assert x in meta and x not in seq and uid(x) not in {uid(s) for s in seq}
        gc=Counter(group(s) for s in seq)
        assert gc[group(x)]*U<q*ref[group(x)]
        assert gc[group(y)]*U>q*ref[group(y)]
        seq[w['slot']]=x;new=stats(seq)
        assert len(seq)==len(set(seq))==q
        assert new['E_numerator']<old['E_numerator']
        assert new['unique_times']>=old['unique_times']
        assert set(initial['demo_counts'])<=set(new['demo_counts'])
        assert set(initial['endpoint_demos'])<=set(new['endpoint_demos'])
        for k,v in new.items():assert v==w['after'][k],(step,k)
        assert w['after']['radius']<=r['before']['radius']+1e-12
        assert w['after']['mean']<=r['before']['mean']+1e-12
        old=new
    assert seq==r['selected_ids']
    for k,v in stats(seq).items():assert v==r['after'][k],k
    assert r['terminal_search']['coverage_legal']==0
    assert r['after']['radius']<=r['before']['radius']+1e-12
    assert r['after']['mean']<=r['before']['mean']+1e-12
    changed=len(set(r['original_ids'])-set(seq))
    assert changed==q-r['retained_original']
    if not r['swaps']:assert seq==r['original_ids']
    return dict(task=r['task'],N=r['N'],q=q,swaps=len(r['swaps']),changed_members=changed,
                E_before=r['before']['E'],E_after=r['after']['E'],
                unique_before=r['before']['unique_times'],unique_after=r['after']['unique_times'],
                radius_before=r['before']['radius'],radius_after=r['after']['radius'],
                mean_before=r['before']['mean'],mean_after=r['after']['mean'],
                endpoint_windows_before=r['before']['endpoint_windows'],endpoint_windows_after=r['after']['endpoint_windows'],
                endpoint_demos_before=len(r['before']['endpoint_demos']),endpoint_demos_after=len(r['after']['endpoint_demos']),
                seconds=r['total_seconds'])

def main():
    evidence_files=sorted(p for p in E.rglob('*') if p.is_file())
    before={str(p.relative_to(ROOT)):sha(p) for p in evidence_files}
    a=load(E/'ACCEPTANCE.json');assert a['status']=='OFFLINE_PASS' and a['inputs_unchanged']
    geo=load(E/'INDEPENDENT_GEOMETRY.json')
    assert geo['status']=='PASS' and geo['inputs_unchanged'] and geo['metadata_unchanged']
    assert sha(ROOT/'verify_remote.py')==geo['code_sha256']
    for p,h in geo['result_sha256'].items():assert sha(E/Path(p).name)==h
    for k in ['optimizer_calls','model_forwards','checkpoint_loads','environment_calls']:assert a[k]==0
    assert (E/'verified_offline.exit').read_text().strip()=='0'
    assert (E/'failure_attempt1/offline.exit').read_text().strip()=='1'
    for name in ['selection.py','test_selection.py','remote_offline_verified.py']:
        assert sha(ROOT/name)==a['code_hashes'][name],name
    summary=[check(load(E/f'task{t}.json')) for t in range(6,10)]
    assert all(g['all_greedy_choices_reproduced']==s['swaps'] and g['terminal_counts'].get('coverage_legal',0)==0 for g,s in zip(geo['tasks'],summary))
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in evidence_files}
    with (ROOT/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    result=dict(status='OFFLINE_ACCEPTANCE_PASS',input_metadata_unchanged=True,
        effective_task6_8_changes=any(r['changed_members'] for r in summary[:3]),
        training_started=False,evaluation_started=False,subsequent_protocol='deferred by user',
        original_failure_exit=1,verified_offline_exit=0,summary=summary,
        verification_limits='Local identity/composition/support replay; real geometry and exhaustive search verified on server using original FP64 distance. No behavioral efficacy claim.',
        evidence_sha256=before)
    (ROOT/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))

if __name__=='__main__':main()
