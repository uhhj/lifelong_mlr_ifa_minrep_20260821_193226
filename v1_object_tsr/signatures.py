"""BC signatures at one immutable pretrained reference; no optimizer updates."""
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
from .state import rng_state,restore_rng
from .io import save,write,sha


def planned_cache(feature_root,task,seed):
    files=sorted((Path(feature_root)/str(task)).glob('*.pt'))
    assert files
    return random.Random(seed).sample(files,max(1,math.ceil(len(files)*.5)))


def probe_files(feature_root,task,seed):
    files=sorted(planned_cache(feature_root,task,seed))
    rng=random.Random('TSR-signature:%d:%d'%(seed,task))
    return rng.sample(files,100) if len(files)>=100 else rng.choices(files,k=100)


def parameter_manifest(model):
    last=len(model.temporal_transformer.decoder.h)-1
    prefix='temporal_transformer.decoder.h.%d.'%last
    names=[n for n,p in model.named_parameters() if n.startswith(prefix) or n.startswith('policy_head.')]
    assert names and any(n.startswith(prefix) for n in names) and any(n.startswith('policy_head.') for n in names)
    return dict(last_decoder_block=last,signature_parameters=[dict(name=n,shape=list(p.shape),count=p.numel()) for n,p in model.named_parameters() if n in names],
        trainable_parameters=[dict(name=n,shape=list(p.shape),count=p.numel()) for n,p in model.named_parameters() if p.requires_grad])


def model_hash(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def bc_eval(model,data):
    feat=model.temporal_encode(data['feature'].squeeze(1))[:,-1,:]
    return model.policy_head.loss_fn(model.policy_head(feat),data['actions'].squeeze(1))


def compute_signature(algo,task):
    begin=time.monotonic();rng=rng_state();student=getattr(algo.policy,'module',algo.policy)
    student_before=model_hash(student)
    try:
        reference=copy.deepcopy(student)
        from libero.lifelong.utils import torch_load_model
        state,_,_=torch_load_model(algo.cfg.pretrain)
        # Native pretrain includes front-end keys; BriefPolicy has only backend.
        missing,unexpected=reference.load_state_dict(state,strict=False)
        assert not missing,('REFERENCE_MISSING_KEYS',missing)
        reference.eval();reference.zero_grad(set_to_none=True)
        manifest=parameter_manifest(reference);names={r['name'] for r in manifest['signature_parameters']}
        for n,p in reference.named_parameters():p.requires_grad_(n in names)
        params=[p for n,p in reference.named_parameters() if n in names]
        before=model_hash(reference);files=probe_files(algo.cfg.feature_dir,task,algo.cfg.seed)
        acc=torch.zeros(sum(p.numel() for p in params),dtype=torch.float32,device=params[0].device)
        from torch.utils.data._utils.collate import default_collate
        losses=[]
        for j in range(10):
            batch=default_collate([torch.load(p,map_location='cpu') for p in files[j*10:(j+1)*10]])
            batch={k:v.to(params[0].device) for k,v in batch.items()}
            loss=bc_eval(reference,batch);losses.append(float(loss.detach()) if bool(torch.isfinite(loss)) else None)
            grads=torch.autograd.grad(loss,params,allow_unused=True)
            acc.add_(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,grads)]))
        acc.div_(10);norm=float(acc.double().norm());valid=math.isfinite(norm) and norm>1e-12 and bool(torch.isfinite(acc).all())
        unit=acc/(norm+1e-12) if valid else torch.zeros_like(acc)
        assert before==model_hash(reference),'REFERENCE_MUTATED'
        assert student_before==model_hash(student),'STUDENT_MUTATED'
        if torch.cuda.is_available():torch.cuda.synchronize()
        meta=dict(task=task,seed=algo.cfg.seed,valid=valid,norm=norm if math.isfinite(norm) else None,
            invalid_reason=None if valid else 'NONFINITE_OR_ZERO_NORM',epsilon_num=1e-12,
            reference_model_sha256=before,pretrain_sha256=sha(algo.cfg.pretrain),
            parameter_manifest=manifest,probe_files=[str(p) for p in files],unique_windows=len(set(files)),
            planned_cache_files=[p.name for p in sorted(planned_cache(algo.cfg.feature_dir,task,algo.cfg.seed))],
            bc_losses=losses,seconds=time.monotonic()-begin,reference_parameter_bytes=sum(p.numel()*p.element_size() for p in reference.parameters()),
            signature_bytes=acc.numel()*acc.element_size(),reference_unexpected_checkpoint_keys=unexpected,
            trajectory_ids_available=False,trajectory_balance='unavailable in original feature cache; user-approved fixed 100 cache windows')
        result=dict(unit=unit.detach().cpu(),mean_gradient=acc.detach().cpu(),valid=valid,metadata=meta)
        del reference
        return result
    finally:restore_rng(rng)


def ensure_signature(algo,task,current_task):
    assert task<=current_task
    path=algo.root/'signatures'/('task%d.pt'%task)
    if path.exists():
        value=torch.load(path,map_location='cpu')
        assert value['metadata']['pretrain_sha256']==sha(algo.cfg.pretrain)
        return value
    assert task==current_task,'HISTORICAL_SIGNATURE_MISSING'
    value=compute_signature(algo,task)
    frozen=json.loads((algo.report.parent/'PARAMETER_MANIFEST.json').read_text())
    assert value['metadata']['parameter_manifest']==frozen,'SIGNATURE_PARAMETER_MANIFEST_CHANGED'
    save(path,value);write(algo.report/('SIGNATURE_task%d.json'%task),dict(value['metadata'],file=str(path),sha256=sha(path)))
    return value
