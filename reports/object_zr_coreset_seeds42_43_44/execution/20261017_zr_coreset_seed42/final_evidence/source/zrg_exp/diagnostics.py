import time,json
import numpy as np
import torch
from .common import *
from .model import readout
def compare(a,b):
    raw=float((a['raw']-b['raw']).abs().max())
    values={k:float((a[k]-b[k]).abs().max()) for k in ('means','scales','probs')}
    ok=raw<=1e-6 and all(torch.allclose(a[k],b[k],atol=1e-6,rtol=1e-6) for k in values) and torch.equal(a['selected'],b['selected'])
    eps=torch.linspace(-1,1,7,device=a['raw'].device).expand(len(a['raw']),7);ix=torch.arange(len(eps),device=eps.device)
    action=lambda x:x['means'][ix,x['selected']]+x['scales'][ix,x['selected']]*eps
    ok=ok and torch.allclose(action(a),action(b),atol=1e-6,rtol=1e-6)
    return dict(raw_max=raw,distribution_max=values,selected_exact=torch.equal(a['selected'],b['selected']),passed=bool(ok))
@torch.no_grad()
def check(p,task,epoch,full=False):
    start=time.time();state=rng();modes={n:m.training for n,m in p.named_modules()};p.eval();rows=[]
    try:
        choices=json.loads((PROTO/'SAMPLE_IDS.json').read_text());bank=np.load(PROTO/'instruction_bank.npz');lastz=None
        for t in range(6,task+1):
            paths=sorted((ROOT/'buffer').glob(f't{t}__*.pt')) if t<task else [P/'features'/str(t)/n for n in choices['incremental'][str(t)]['probe']]
            if not full:paths=paths[:8]
            for pos in range(0,len(paths),10):
                h=torch.cat([torch.load(v,map_location='cpu')['feature'] for v in paths[pos:pos+10]]).cuda();z=p.temporal_encode(h)[:,-1];e=h[:,-1,0];lastz=z
                a=readout(p,z,e);disabled=[n for n in p.residual_bank if int(n)>t]
                b=readout(p,z,e,disabled);cmp=compare(a,b)
                contrib=sum((p.residual_bank[n](z,e) for n in disabled),torch.zeros(len(z),75,device=z.device))
                old=max((float(p.residual_bank[n].gate(e).abs().max()) for n in disabled),default=0.)
                cur=float((p.residual_bank[str(t)].gate(e)-1).abs().max())
                row=dict(kind='legal_buffer' if t<task else 'current_fixed_probe',task=t,n=len(z),old_gate=old,current_error=cur,contribution=float(contrib.abs().max()),**cmp)
                row['passed']=row['passed'] and old<=1e-6 and cur<=1e-5 and row['contribution']<=1e-6;rows.append(row)
        # Previously approved real language variants, same live shared z; no base-task action training.
        for t in range(task+1):
            mask=(bank['variant_task_ids']==t)&~np.char.startswith(bank['variant_source'],'train_mode')
            ev=torch.from_numpy(bank['variant_embeddings'][mask]).cuda();z=lastz[:1].expand(len(ev),-1)
            disabled=[n for n in p.residual_bank if int(n)>t]
            a=readout(p,z,ev);b=readout(p,z,ev,disabled);cmp=compare(a,b)
            contrib=sum((p.residual_bank[n](z,ev) for n in disabled),torch.zeros(len(z),75,device=z.device))
            old=max((float(p.residual_bank[n].gate(ev).abs().max()) for n in disabled),default=0.)
            row=dict(kind='approved_language_variants_live_z',task=t,n=len(ev),old_gate=old,contribution=float(contrib.abs().max()),**cmp)
            row['passed']=row['passed'] and old<=1e-6 and row['contribution']<=1e-6;rows.append(row)
        result=dict(stage=task,epoch=epoch,full_legal_old_buffer=full,rows=rows,passed=all(r['passed'] for r in rows),seconds=time.time()-start,branch_norms={n:dict(U=float(ch.U.norm()),V=float(ch.V.norm())) for n,ch in p.residual_bank.items()})
        append(O/'DIRECT_CHANNEL_ISOLATION.jsonl',result)
        assert result['passed'],'STRUCTURAL_NUMERICAL_FAILURE: preserve result; do not loosen tolerances'
    finally:
        for n,m in p.named_modules():m.training=modes[n]
        restore_rng(state)
    return result
