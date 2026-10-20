"""Local independent outcome/lineage/KL audit. Only stdlib and existing metadata."""
import json, hashlib, sys, time
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
from flow_metrics import metrics, pack
def main(D):
    E=D/'evidence';manifest=json.loads((E/'PACKAGE_MANIFEST.json').read_text())
    for n,x in manifest.items():
        raw=(E/n).read_bytes();assert len(raw)==x['bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256']
    pm,pr,pairs,ph=metrics(E/'controls/RA',41,'ra_coreset')
    vm,vr,vpairs,vh=metrics(E/'controls/ZR_v2',41,'ra_coreset_ifa_zrv2')
    assert (pm['AUC'],pm['FWT'],pm['OldFinal'])==(F(701,960),F(3,4),F(47,60))
    results={}
    for variant in ('v3','v3a'):
        root=E/variant
        if not (root/'PROCESS_EXIT_ACCEPTANCE.json').exists():continue
        js=lambda p:json.loads(p.read_text())
        a=js(root/'PROCESS_EXIT_ACCEPTANCE.json');assert a['status']=='PASS'
        spec=js(root/'RUN_SPEC.json');arm=spec['arm']
        m,r,rows,hs=metrics(root/'evaluation',41,arm)
        for k in rows:
            assert rows[k]['parent_checkpoint_sha256']==pairs[k]['checkpoint_sha256']
            for f in ('modal_sha256','initial_state_sha256','initial_state_index'):assert rows[k][f]==pairs[k][f]
        events=Counter(json.loads(l)['event'] for p in (root/'evaluation').glob('attempts*.jsonl') for l in p.read_text().splitlines())
        assert events==Counter(vector_attempt_start=20,vector_attempt_complete=20,environment_construct_start=10,environment_construct_complete=10,environment_closed=10)
        ep=js(root/'epochs.json');assert len(ep)==40 and sum(x['batches'] for x in ep)==10550 and sum(x['samples'] for x in ep)==105240
        st=[json.loads(l) for l in (root/'STRUCTURE.jsonl').read_text().splitlines()]
        assert len(st)==16 and all(x['passed'] and x['zero_residual_parent_exact'] and x['unselected_means_exact'] for x in st)
        assert js(root/'CHECKPOINT_LOAD_ACCEPTANCE.json')['status']=='PASS' and len(js(root/'CHECKPOINT_LOAD_ACCEPTANCE.json')['rows'])==4
        cells=[js(p) for p in (root/'evaluation').glob('cell*.json')];assert len(cells)==10
        if variant=='v3':assert max(x['actual_kl_max'] for x in st)<=.5+1e-12 and max(x['max_executed_kl'] for x in cells)<=.5+1e-12
        for n in ('pipeline','calibrate','evaluate','audit'):assert (root/(n+'.exit')).read_text().strip()=='0'
        gate=m['AUC']>pm['AUC'] and m['FWT']>=pm['FWT'] and m['OldFinal']>=pm['OldFinal']
        assert gate==a['gate_passed'] and pack(m)==a['new']
        results[variant]=dict(metrics=pack(m),parent_delta=pack({k:m[k]-pm[k] for k in m}),v2_delta=pack({k:m[k]-vm[k] for k in m}),gate=gate,
                              matrix=[[float(r[s,t]*100) if t<=s else None for t in range(6,10)] for s in range(6,10)],
                              cache_max_KL=max(x['actual_kl_max'] for x in st),online_max_KL=max(x['max_executed_kl'] for x in cells),
                              projected_online_states=sum(x['projected_states'] for x in cells),online_states=sum(x['sampled_states'] for x in cells))
    complete=(E/'current/PROCESS_EXIT_ACCEPTANCE.json').exists()
    if complete:
        assert json.loads((E/'current/PROCESS_EXIT_ACCEPTANCE.json').read_text())['status']=='PASS'
        assert (E/'current/controller.exit').read_text().strip()=='0' and (E/'current/supervisor.exit').read_text().strip()=='0'
        assert 'v3' in results
        assert ('v3a' in results)==results['v3']['gate']
        assert not json.loads((D/'SNAPSHOT.json').read_text())['workers']
    out=dict(status='PASS_FINAL' if complete else 'PASS_PARTIAL',time=time.time(),parent=pack(pm),old_v2=pack(vm),variants=results,
             package_files=len(manifest),new_training_or_rollouts=False)
    (D/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2));return out
if __name__=='__main__':main(Path(sys.argv[1]))
