"""CPU-only import/config/actual-completed-checkpoint compatibility admission."""
from pathlib import Path
import ast,contextlib,copy,hashlib,io,json,os,sys,time
Q=Path(__file__).resolve().parent.parent
case=sys.argv[1];R=Q/'cases'/case
J=lambda p:json.loads(Path(p).read_text())
inp=J(R/'INPUT.json');assert inp['method']=='MLR'
os.environ.update(REVERSE_CASE_ROOT=str(R),CORESET_OUT=inp['input_root'],CORESET_SEED=str(inp['seed']),CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path.insert(0,str(Q));sys.path.insert(0,str(R/'source'));sys.path.insert(0,str(Q/'evaluation_unification_v1'))
from order import install
install()
from reverse_common import verify_case,sha
contract=verify_case(R)
tic=time.time()
with contextlib.redirect_stdout(io.StringIO()):
    import unified_mlr_runtime as rt
    import evaluate
    from libero.lifelong.models.brief import BriefPolicy
    from easydict import EasyDict
    import torch
assert Path(evaluate.__file__).resolve()==Q/'evaluation_unification_v1/evaluate.py'
assert not torch.cuda.is_initialized()
manifest=J(Q/'EVALUATION_PROTOCOL_AMENDMENT.json')
assert manifest['cases'][case]['training_contract_sha256']==contract
for p,h in manifest['evaluation_source_sha256'].items():assert sha(Q/p)==h
peer=J(Q/'cases'/manifest['cases'][case]['peer_DEBR_case']/'INPUT.json')
ignored={'buffer_dir','contract','experiment_dir','feature_dir','run_root','selector','shape_meta'}
assert {k:v for k,v in inp['eval_config'].items() if k not in ignored}=={k:v for k,v in peer['eval_config'].items() if k not in ignored}
architecture={}
for name in ['models/brief.py','models/complete_policy.py','models/policy_head.py','models/modules/transformer_modules.py']:
    a=R/'source/libero/lifelong'/name;b=Path(peer['source_root'])/'libero/lifelong'/name
    exact=ast.dump(ast.parse(a.read_text()),include_attributes=False)==ast.dump(ast.parse(b.read_text()),include_attributes=False)
    assert exact,(case,name,'Architecture AST differs')
    architecture[name]=dict(AST_equal=True,MLR_sha256=sha(a),DEBR_sha256=sha(b))
cfg=EasyDict(copy.deepcopy(inp['eval_config']));cfg.device='cpu';cfg.training=False
models={};forwards=0
for stage in range(6,10):
    path=rt.model_path('MLR',stage)
    if not path.exists():continue
    raw=torch.load(path,map_location='cpu')['state_dict']
    state={(k[7:] if k.startswith('module.') else k):v for k,v in raw.items()}
    with contextlib.redirect_stdout(io.StringIO()):m=BriefPolicy(copy.deepcopy(cfg),None)
    m.eval().requires_grad_(False)
    assert len(m.state_dict())==len(state)==100
    m.load_state_dict(state,strict=True)
    assert all(torch.equal(v,state[k]) for k,v in m.state_dict().items())
    models[str(stage)]=dict(sha256=sha(path),strict_keys=100,parameters=sum(v.numel() for v in m.parameters()),actual_saved_MLR_model_loaded=True)
    if stage==6:
        man=J(inp['manifests'][str(stage)]['path']);sample=torch.load(man['rows'][0]['path'],map_location='cpu')['feature']
        if sample.ndim==5:sample=sample.squeeze(1)
        with torch.no_grad():
            for length in range(1,9):
                y=m.policy_head.forward_fn(m.temporal_encode(sample[:,:length])[:,-1])
                assert all(torch.isfinite(v).all() for v in y) and bool((y[1]>0).all())
                forwards+=1
    del m,state,raw
assert not torch.cuda.is_initialized() and verify_case(R)==contract
print(json.dumps(dict(status='PASS_CPU_IMPORT_CONFIG_AND_CHECKPOINT_COMPATIBILITY',case=case,training_contract_sha256=contract,scientific_eval_config_equal=True,architecture=architecture,completed_checkpoints=models,CPU_policy_forward_calls=forwards,CUDA_initialized=False,new_optimizer=0,new_environment=0,formal_cache_parity='PENDING_OWN_GPU_ADMISSION_BEFORE_FIRST_FORMAL_EPISODE',seconds=time.time()-tic)))
