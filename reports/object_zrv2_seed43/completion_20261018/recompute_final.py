"""Read-only final evidence verification; no model, optimizer or environment loading."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter, defaultdict
import hashlib, json, sys

P = Path(__file__).resolve().parent
R = P.parent
E = P / 'evidence'
sys.dont_write_bytecode = True
sys.path.insert(0, str(R / 'attachment/zr_v2_seed43_plan'))
from compare_metrics import compare, load_records

def js(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def exact(r):
    return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/F(10-t) for t in range(6,10))/4,
                FWT=sum(r[t,t] for t in range(6,10))/4,
                NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/F(9-t) for t in range(6,9))/3,
                OldFinal=sum(r[9,t] for t in range(6,9))/3,
                FinalAll=sum(r[9,t] for t in range(6,10))/4)
def main():
    manifest=js(E/'PACKAGE_MANIFEST.json')
    for n,v in manifest['files'].items():
        p=E/n; assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],n
    contract=js(E/'CONTRACT.json')
    for p in (R/'source').rglob('*.py'):
        remote='/root/mlr_ifa/experiment_workspaces/ZRv2_OBJECT_seed43_20261018/source/'+p.relative_to(R/'source').as_posix()
        assert sha(p)==contract['files'][remote],remote
    comparison=compare(R/'parent_evaluation',E/'evaluation')
    assert comparison==js(E/'METRIC_COMPARISON.json')
    br,b,_=load_records(R/'parent_evaluation',False)
    nr,n,_=load_records(E/'evaluation',True)
    bm,nm=exact(br),exact(nr)
    assert all(str(v)==comparison['new'][k]['fraction'] for k,v in nm.items())
    binding=js(E/'PARENT_BINDING.json')
    for name,h in binding['episode_hashes'].items(): assert sha(R/'parent_evaluation'/name)==h
    epochs=js(E/'epochs.json')
    assert {(r['task'],r['epoch']) for r in epochs}=={(t,e) for t in range(6,10) for e in range(10)} and len(epochs)==40
    assert sum(r['batches'] for r in epochs)==10550 and sum(r['samples'] for r in epochs)==105240
    batches=rows(E/'calibration_batches.jsonl');assert len(batches)==10550 and len({r['logical_step'] for r in batches})==10550
    index=defaultdict(list)
    for r in batches:
        assert len(r['indices'])==len(r['samples'])
        index[r['stage'],r['epoch']].extend(r['indices'])
    for (t,e),v in index.items(): assert sorted(v)==list(range([1562,2261,3081,3620][t-6]))
    for t in range(6,10):
        last=next(r for r in epochs if r['task']==t and r['epoch']==9)
        for j in range(6,t+1): assert last['branch_steps'][str(j)]==sum(j in r['active'] for r in batches if r['stage']==t)
    structures=rows(E/'STRUCTURE.jsonl');assert len(structures)==16 and all(r['passed'] for r in structures)
    assert len({r['T0_hash'] for r in structures})==1
    attempts=[r for p in (E/'evaluation').glob('attempts*.jsonl') for r in rows(p)]
    events=Counter(r['event'] for r in attempts)
    assert events==Counter(vector_attempt_start=20,vector_attempt_complete=20,environment_construct_start=10,environment_construct_complete=10,environment_closed=10)
    groups={(r['stage'],r['task'],r['group']):r for r in attempts if r['event']=='vector_attempt_complete'}
    assert len(groups)==20
    for key,r in n.items():
        s,t,i=key;assert r['success']==groups[s,t,i//10]['successes'][i%10]
    cps=js(E/'CHECKPOINT_LOAD_ACCEPTANCE.json');assert cps['status']=='PASS' and {r['stage'] for r in cps['rows']}==set(range(6,10))
    for cp in cps['rows']:
        assert {r['checkpoint_sha256'] for r in n.values() if r['stage']==cp['stage']}=={cp['checkpoint_sha256']}
    audit=js(E/'AUDIT.json');exit_audit=js(E/'PROCESS_EXIT_ACCEPTANCE.json')
    assert audit['status']==exit_audit['status']=='PASS' and exit_audit['no_remaining_processes']
    for name in ['calibrate','evaluate','audit','pipeline']:assert (E/(name+'.exit')).read_text().strip()=='0'
    recovery=E/'engineering_recovery/20261018_final_path'
    assert (recovery/'original_failure.exit').read_text().strip()=='1' and (recovery/'fixed_audit.exit').read_text().strip()=='0'
    rec=js(recovery/'RECOVERY_ADDENDUM.json')
    assert sha(P/'engineering_recovery/final_process_audit_fixed.py')==rec['fixed_sha256']
    assert sha(P/'engineering_recovery/final_process_audit.original.py')==rec['original_sha256']
    contributions=[]
    for t in range(6,10):
        ac=(sum(nr[s,t]-br[s,t] for s in range(t,10))/F(10-t))/4
        nb=(sum((nr[t,t]-nr[s,t])-(br[t,t]-br[s,t]) for s in range(t+1,10))/F(9-t))/3 if t<9 else F(0)
        contributions.append(dict(task=t,auc_delta_pp=float(ac*100),nbt_delta_pp=float(nb*100)))
    out=dict(status='PASS',engineering='COMPLETED',performance_status='ELIGIBLE_FOR_REVIEW' if comparison['metric_gate'] else 'AUDIT_PASS_METRIC_NOT_PASSED',
             parent='seed43_repeat1_Coreset_IFA',baseline_records=len(b),new_records=len(n),checked_package_files=len(manifest['files']),epochs=40,optimizer_steps=10550,exposures=105240,checkpoints=4,structure_checks=16,attempt_events=dict(events),
             comparison=comparison,matrix=[dict(stage=s,task=t,baseline_percent=float(br[s,t]*100),new_percent=float(nr[s,t]*100),delta_pp=float((nr[s,t]-br[s,t])*100)) for s,t in sorted(br)],
             metric_contributions=contributions,timings=rows(E/'timings.jsonl'),extension_started=False,decision='STOP: all four predeclared criteria are required; NBT failed.' if not comparison['metric_gate'] else 'All criteria passed; extension authorized.',
             limitation='Local audit verifies downloaded records and remote checkpoint-load attestations; no checkpoint was downloaded or reloaded locally.')
    (P/'LOCAL_FINAL_ACCEPTANCE.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:out[k] for k in ['status','engineering','performance_status','new_records','comparison','metric_contributions','timings','decision']}))

if __name__=='__main__':main()
