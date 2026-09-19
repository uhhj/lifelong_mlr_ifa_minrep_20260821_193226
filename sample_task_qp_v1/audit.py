"""No CUDA/model construction. Verify original serialized state bytes and data."""
import gzip,time,subprocess
from .common import *
def run():
    for n,h in read(OUT/'FROZEN_INPUTS.json').items():assert sha(ROOT/n)==h,n
    p=read(OUT/'PROTOCOL_FROZEN.json');old=read(OLD/'ASSET_AUDIT.json');states={x['point']:x for x in old['states']};checked=[]
    for name in p['points']:
        r=states[name];f=Path(r['path']);assert f.is_file() and sha(f)==r['sha256'],name
        assert {'model','optimizer','scheduler','rng','optimizer_step','scheduler_step'}<=set(r['state_keys'])
        checked.append(dict(point=name,path=str(f),sha256=r['sha256'],bytes=f.stat().st_size))
    with gzip.open(ROOT/'reports/gradient_relation_4090_v2/DESIGN.json.gz','rt') as f:design=json.load(f)
    for spec in design['source_plan_sha256'].values():assert sha(ROOT/spec['path'])==spec['sha256']
    records=read(OUT/'PROBE_WINDOW_MEMBERSHIP.json')['windows'];counts={};seen={}
    for r in records:
        if r['point'] not in p['points']:continue
        assert sha(ROOT/r['path'])==r['sha256'],r['path']
        if r['eval_task']<r['current_task']:assert r['source']=='actual_immutable_buffer'
        k=(r['point'],r['eval_task']);counts[k]=counts.get(k,0)+1;seen.setdefault(k,set()).add(r['demo'])
    assert len(checked)==36 and all(n==32 for n in counts.values())
    artifact_counts={}
    for name in p['points']:
        dest=OLD/'measurements'/name;complete=read(dest/'COMPLETE.json')
        for fn,h in complete['files'].items():assert sha(dest/fn)==h
        decisions=rows(dest/'DECISIONS_FROZEN.jsonl')
        first=[r for r in decisions if r['step']==1]
        assert {r['arm'] for r in first}=={'TASK_GRAD_QP','TASK_STEP_QP'}
        artifact_counts[name]=len(first)
    write(OUT/'ASSET_AUDIT.json',dict(status='CPU_ASSET_AUDIT_PASS',states=checked,
      window_groups=[dict(point=k[0],task=k[1],windows=n,demos=len(seen[k])) for k,n in counts.items()],
      original_first_step_task_QP_available=artifact_counts,per_sample_gradients_not_recoverable_from_task_Gram=True,
      gpu_required=True,cpu_only=True,no_new_carrier=True,restore_note='Byte-identical serialized originals verified; live exact restoration and native step still gated on GPU.',
      git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),time=time.time()))
    print('CPU_ASSET_AUDIT_PASS',len(checked),len(records))
if __name__=='__main__':run()
