"""Fixed-checkpoint reference comparison; no optimizer or file mutations."""
import sys, json, random, re, math, contextlib
from pathlib import Path
import torch
from easydict import EasyDict
from torch.utils.data import default_collate
from torch.utils.data.distributed import DistributedSampler
ROOT=Path('/media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep')
sys.path.insert(0,str(ROOT))
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.algos.mlr import build_ifa_similarity_matrix
from libero.lifelong.algos.base import l2norm
torch.set_num_threads(2)
rows=[]
for seed,stage in ((42,7),(42,8),(43,7),(44,7)):
    run=ROOT/f'runs/final_paper_text_faithful/ifa/seed{seed}/run_001'
    cfg=EasyDict(json.loads((run/'config.json').read_text()))
    feature=Path(cfg.feature_dir); buffer=Path(cfg.buffer_dir)
    replay=sorted(str(p) for p in buffer.glob('*.pt') if int(re.match(r't(\d+)__',p.name)[1])<stage)
    current=sorted(str(p) for p in (feature/str(stage)).glob('*.pt'))
    # Step 1 established append-only replay and exact final-buffer equality.
    mixed=replay+current; random.Random(seed).shuffle(mixed)
    ids=[int(re.match(r't(\d+)__',Path(p).name)[1]) if Path(p).name.startswith('t') else stage for p in mixed]
    with contextlib.redirect_stdout(sys.stderr):
        info=build_ifa_similarity_matrix(stage,str(feature),str(buffer))
    old=[j if k==stage else k for j,k in info['selected_pairs']]
    assert len(old)==1
    negative=old[0]
    cos_ab=info['sim_language'][info['task_ids'].index(stage),info['task_ids'].index(negative)]
    margin=.30*torch.acos(torch.tensor(float(cos_ab),device='cuda').clamp(-1+1e-6,1-1e-6))
    neg=torch.load(feature/str(negative)/'0.pt',map_location='cpu')['feature'][0,0,0,:].float().cuda()
    with contextlib.redirect_stdout(sys.stderr):
        model=BriefPolicy(cfg,cfg.shape_meta).cuda()
    model.train()
    sampler=DistributedSampler(ids,num_replicas=1,rank=0,shuffle=True)
    for epoch in (70,99):
        ckpt=run/f'mlrifa_{stage}_model_ep{epoch}.pth'
        checkpoint=torch.load(ckpt,map_location='cpu')
        state={k.removeprefix('module.') if hasattr(k,'removeprefix') else (k[7:] if k.startswith('module.') else k):v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state,strict=True)
        del state,checkpoint
        sampler.set_epoch(epoch); order=list(sampler); chosen={}
        for offset in range(0,len(order),10):
            ix=order[offset:offset+10]; tasks=[ids[i] for i in ix]
            if stage not in tasks: continue
            category='correct' if tasks[0]==stage else ('negative_task' if tasks[0]==negative else 'other_old_task')
            if chosen.get(category,0)>=3: continue
            chosen[category]=chosen.get(category,0)+1
            data=default_collate([torch.load(mixed[i],map_location='cpu') for i in ix])
            data={k:v.cuda() for k,v in data.items()}
            mask=torch.tensor(tasks,device='cuda')==stage
            torch.manual_seed(100000+seed*1000+epoch*10+offset); torch.cuda.manual_seed_all(100000+seed*1000+epoch*10+offset)
            with torch.no_grad():
                bc,latent=model.compute_loss(data,None,None)
            g=latent[mask].detach().requires_grad_(True)
            f=data['feature'].squeeze(1).float()
            wrong=f[0,0,0,:].expand(int(mask.sum()),-1)
            right=f[mask,0,0,:]
            losses=[]; grads=[]; active=[]
            for ref in (wrong,right):
                # Match release's repeated normalization exactly.
                gn=l2norm(l2norm(g)); rn=l2norm(l2norm(ref)); nn=l2norm(l2norm(neg.expand_as(ref)))
                h=torch.relu(torch.acos((gn*rn).sum(-1).clamp(-1+1e-6,1-1e-6))-torch.acos((gn*nn).sum(-1).clamp(-1+1e-6,1-1e-6))+margin)
                loss=.1*h.mean()
                grad=torch.autograd.grad(loss,g)[0]
                assert torch.isfinite(loss) and torch.isfinite(grad).all()
                losses.append(float(loss)); grads.append(grad); active.append(float((h>0).float().mean()))
            norms=[float(v.norm()) for v in grads]
            cosine=float(torch.nn.functional.cosine_similarity(grads[0].flatten(),grads[1].flatten(),dim=0)) if min(norms)>1e-10 else None
            rows.append(dict(seed=seed,stage=stage,checkpoint_epoch=epoch,batch_index=offset//10,category=category,tasks=tasks,bc_loss=float(bc),margin=float(margin),own_reference_max_abs_difference=float((wrong-right).abs().max()),release_reference_vs_negative_max_abs=float((wrong-neg).abs().max()),weighted_ifa_release=losses[0],weighted_ifa_corrected=losses[1],active_release=active[0],active_corrected=active[1],g_gradient_norm_release=norms[0],g_gradient_norm_corrected=norms[1],g_gradient_cosine=cosine))
            expected={'correct','negative_task'} | ({'other_old_task'} if stage==8 else set())
            if all(chosen.get(c,0)==3 for c in expected): break
    del model
    torch.cuda.empty_cache()
print(json.dumps({'scope':'same train-mode forward and dropout realization shared by both losses; autograd only with respect to detached global latent g; no parameter gradients or optimizer updates','sampling':'first three eligible batches per source category, sampler epochs70/99; evaluated at corresponding END-of-epoch checkpoints, not historical within-epoch model states','rows':rows},indent=2))
