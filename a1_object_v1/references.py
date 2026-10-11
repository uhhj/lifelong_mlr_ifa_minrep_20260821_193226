"""Only pretrained task descriptions and already-arrived cache references."""
import hashlib
import time
from pathlib import Path
import torch
from .geometry import append_basis, validate
from .state import rng_state, restore_rng
from .io import save, write, sha


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def recover_base(cfg):
    from .bindings import bind
    bind()
    from easydict import EasyDict
    from peft import LoraConfig, get_peft_model
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.models.modules.language_modules import CLIPTextEncoder
    from libero.lifelong.utils import get_task_embs, torch_load_model
    begin=time.monotonic(); rng=rng_state()
    try:
        # Same language-only construction as CompletePolicy; load every base,
        # adapter and projection key from the actual pretrained checkpoint.
        kw=dict(cfg.policy.language_encoder.network_kwargs);kw['output_size']=cfg.policy.embed_size
        enc=get_peft_model(CLIPTextEncoder(**kw),LoraConfig(r=cfg.policy.rank,
            lora_alpha=cfg.policy.alpha,target_modules=['q_proj','v_proj']))
        state,_,_=torch_load_model(cfg.pretrain)
        weights={k[len('language_encoder.'):]:v for k,v in state.items() if k.startswith('language_encoder.')}
        assert weights, 'PRETRAIN_LANGUAGE_WEIGHTS_MISSING'
        enc.load_state_dict(weights,strict=True);enc.cuda().eval()
        benchmark=get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
        descriptions=[benchmark.get_task(i).language for i in range(7)]
        tokens=get_task_embs(cfg,descriptions)
        with torch.no_grad(): encoded=enc(**{k:v.cuda() for k,v in tokens.items()}).cpu()
        assert encoded.shape==(7,512)
        cached=torch.load(Path(cfg.feature_dir)/'6/0.pt',map_location='cpu')['feature'][0,0,0,:]
        direction_error=float((encoded[6].double()/encoded[6].double().norm()-cached.double()/cached.double().norm()).norm())
        assert direction_error <= 1e-5, ('PRETRAIN_CACHE_LANGUAGE_MISMATCH',direction_error)
        refs=[r.clone() for r in encoded[:6]]
        U=torch.empty(512,0,dtype=torch.float64)
        for r in refs:U,_=append_basis(U,r)
        meta=dict(task_ids=list(range(6)),descriptions=descriptions[:6],
            reference_source='CompletePolicy.language_encoder after pretrained base+LoRA strict load',
            checkpoint=cfg.pretrain,checkpoint_sha256=sha(cfg.pretrain),language_state_keys=len(weights),
            validation_task=6,validation_cache=str(Path(cfg.feature_dir)/'6/0.pt'),
            cache_direction_error=direction_error,errors=validate(U,refs),seconds=time.monotonic()-begin,
            future_reference_access=False,encoder_source='libero/lifelong/models/complete_policy.py:73-76')
        return dict(refs=refs,U64=U,U32=U.float(),task_ids=list(range(6)),metadata=meta)
    finally:
        restore_rng(rng)


def ensure_stage(algo,task):
    begin=time.monotonic(); rng=rng_state()
    directory=algo.root/'subspaces';directory.mkdir(parents=True,exist_ok=True)
    path=directory/('SUBSPACE_stage%d.pt'%task)
    try:
        if path.exists():
            state=torch.load(path,map_location='cpu')
            assert state['task_ids']==list(range(task+1))
        else:
            parent=directory/('SUBSPACE_stage%d.pt'%(task-1)) if task>6 else Path(algo.cfg.a1_base_reference)
            state=torch.load(parent,map_location='cpu')
            assert state['task_ids']==list(range(task))
            source=Path(algo.cfg.feature_dir)/str(task)/'0.pt'
            r=torch.load(source,map_location='cpu')['feature'][0,0,0,:].clone()
            U,residual=append_basis(state['U64'],r)
            state=dict(refs=state['refs']+[r],U64=U,U32=U.float(),task_ids=list(range(task+1)),
                metadata=dict(parent=str(parent),parent_sha256=sha(parent),source=str(source),source_sha256=sha(source),
                    reference_sha256=tensor_sha(r),append_residual=residual,stage=task,seconds=time.monotonic()-begin))
            state['metadata']['errors']=validate(U,state['refs'])
            save(path,state)
        validate(state['U64'],state['refs'])
        assert state['U64'].shape[1] <= task+1
        algo.a1_U=state['U32'].to(algo.cfg.device).detach()
        algo.a1_subspace_state=state
        write(algo.report/('SUBSPACE_stage%d.json'%task),dict(state['metadata'],task_ids=state['task_ids'],
            rank=state['U64'].shape[1],path=str(path),sha256=sha(path),
            tensor_bytes=sum(x.numel()*x.element_size() for x in [state['U64'],state['U32']]+state['refs'])))
    finally:
        restore_rng(rng)
