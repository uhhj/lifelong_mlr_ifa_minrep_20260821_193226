"""Alternative retaining full-window FP32 operations; offline, no environment."""
import sys,json,time,hashlib,torch
from pathlib import Path
torch.cuda.set_per_process_memory_fraction(0.30)
torch.backends.cudnn.benchmark=True
from coreset_exp.model import cfg,data_setup,complete,strict_downstream
from coreset_exp.common import O,sha
from libero.lifelong.datasets import get_dataset
from libero.lifelong.utils import control_seed
def cmp(a,b):return dict(equal=bool(torch.equal(a,b)),max_abs=float((a-b).abs().max()),different=int(torch.count_nonzero(a!=b)))
def main():
 control_seed(809111);c=cfg('eval');b,_,shape=data_setup(c,False);c.shape_meta=shape
 p=complete(c,shape).eval().requires_grad_(False)
 ds,_=get_dataset(str(Path(c.folder)/b.get_task_demonstration(6)),c.data.obs.modality,False,seq_len=8,demo_limit=10,train=False)
 instr,mask=b.get_task_emb(6)
 def data(offset):
  rows=[ds[min(i*31+offset,len(ds)-1)] for i in range(10)]
  return dict(obs={k:torch.stack([torch.as_tensor(r['obs'][k]) for r in rows]).cuda() for k in shape['all_shapes']},task_emb=instr.repeat(10,1).cuda(),masks=mask.repeat(10,1).cuda())
 d=data(0);other=data(8)
 def run():return p.temporal_encode(p.spatial_encode(p.preprocess_input(d,train_mode=False)))[:,-1]
 def timed(fn):
  torch.cuda.synchronize();t=time.perf_counter();out=fn();torch.cuda.synchronize();return out,time.perf_counter()-t
 rows=[]
 with torch.no_grad():
  # Both checkpoints: graph captures identical operations, never modifies weights.
  for stage in [6,9]:
   cp=O/'random/models'/f'mlrifa_{stage}_model_ep99.pth';strict_downstream(p,cp)
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):run()
   torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out=run()
   tests=[]
   for offset in [0,8]:
    new=data(offset)
    for k in d['obs']:d['obs'][k].copy_(new['obs'][k])
    ref,_=timed(run);graph.replay();torch.cuda.synchronize()
    rng=torch.cuda.get_rng_state();a=p.policy_head(ref,train=False).sample();r1=torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(rng);a2=p.policy_head(out,train=False).sample();r2=torch.cuda.get_rng_state()
    tests.append(dict(offset=offset,features=cmp(ref,out),actions=cmp(a,a2),rng_equal=bool(torch.equal(r1,r2))))
   pairs=[]
   for i in range(8):
    _,t0=timed(run);_,t1=timed(graph.replay);pairs.append([t0,t1])
   rows.append(dict(stage=stage,checkpoint_sha256=sha(cp),tests=tests,seconds_original_graph=pairs))
   del graph,out
 print('RESULT_JSON='+json.dumps(dict(status='OFFLINE_GRAPH_PROBE_ONLY_NOT_DEPLOYED',rows=rows,peak_allocated=torch.cuda.max_memory_allocated(),new_optimizer=0,new_environment=0,production_writes=0,shared_gpu=True)))
if __name__=='__main__':main()
