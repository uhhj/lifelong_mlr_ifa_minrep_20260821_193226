import pickle
from runtime import *
def initialization(a,arm):
 r=rng();digest={k:hashlib.sha256(pickle.dumps(v) if k in ('python','numpy') else b''.join(x.cpu().numpy().tobytes() for x in (v if isinstance(v,list) else [v]))).hexdigest() for k,v in r.items()}
 tensor={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in a.policy.state_dict().items()}
 expected=js(O/'ra_coreset/INITIAL_STATE.json')
 assert tensor==expected['tensors'] and all(digest[k]==expected['rng'][k] for k in digest if k!='cuda'),'Official initialization/non-CUDA RNG mismatch'
 props=torch.cuda.get_device_properties(0)
 hardware=dict(device=props.name,capability=[props.major,props.minor],total_memory=props.total_memory,torch_version=torch.__version__,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)
 write(R/('INITIAL_'+arm+'.json'),dict(status='PASS_OFFICIAL_INITIALIZATION',tensors=tensor,rng=digest,hardware=hardware,historical_cuda_rng_equal=digest['cuda']==expected['rng']['cuda'],historical_cuda_rng_gate=False,official_cp_sha256=sha(a.cfg.pretrain),parent_initial_sha256=sha(O/'ra_coreset/INITIAL_STATE.json')))
