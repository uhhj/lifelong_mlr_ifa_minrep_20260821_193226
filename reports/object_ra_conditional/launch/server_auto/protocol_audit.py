"""Protocol equivalence admission; read only, no model forward/optimizer/environment."""
import os,sys,json,ast,hashlib,time
from pathlib import Path
import torch
Q=Path('/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019')
N=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def flat(x,p=''):
 if isinstance(x,dict):return {k:v for n,y in x.items() for k,v in flat(y,p+'.'+str(n)).items()}
 return {p:x}
def main():
 from coreset_exp.common import O,S,SEED,verify,rng
 from libero.lifelong.algos.mlr import build_dataset_for_task
 from torch.utils.data import DistributedSampler
 c=verify();P=Path(os.environ['RA_PARENT']);PS=Path(os.environ['RA_PARENT_SOURCE'])
 allowed={'.contract','.selector','.buffer_dir','.feature_dir','.experiment_dir','.run_root'}
 a=flat(js(O/'resolved_ra_coreset.json'));b=flat(js(P/'resolved_coreset.json'))
 diffs={k:dict(current=a.get(k),parent=b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
 assert set(diffs)<=allowed,diffs
 def methods(p):
  return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 aa=methods(S/'coreset_exp/train.py');bb=methods(PS/'coreset_exp/train.py')
 # Seed42 parent literals are equivalent to SEED=42, otherwise exact AST.
 if SEED==42:
  text=(PS/'coreset_exp/train.py').read_text().replace('control_seed(42)','control_seed(SEED)').replace('c.buffer_dir,42,start=6','c.buffer_dir,SEED,start=6')
  bb={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 same={n:aa[n]==bb[n] for n in ['references','PaperMLR','build','capture','restore']};assert all(same.values()),same
 critical=['libero/lifelong/algos/mlr.py','libero/lifelong/algos/base.py','libero/lifelong/datasets.py','libero/lifelong/utils.py']
 critical += [str(p.relative_to(S)) for p in (S/'libero/lifelong/models').rglob('*.py')]
 source={n:dict(current=sha(S/n),parent=sha(PS/n)) for n in critical}
 assert all(v['current']==v['parent'] for v in source.values())
 before=torch.get_rng_state().clone()
 da=build_dataset_for_task(str(O/'features'),6,str(O/'ra_coreset/buffer'),SEED,start=6)
 db=build_dataset_for_task(str(P/'features'),6,str(P/'coreset/buffer'),SEED,start=6)
 assert len(da)==len(db)==1562
 ordered_a=[(Path(p).name,sha(Path(p))) for p in da.file_paths]
 ordered_b=[(Path(p).name,sha(Path(p))) for p in db.file_paths];assert ordered_a==ordered_b
 hashes=[]
 sa=DistributedSampler(da,num_replicas=1,rank=0,shuffle=True);sb=DistributedSampler(db,num_replicas=1,rank=0,shuffle=True)
 for e in range(100):
  sa.set_epoch(e);sb.set_epoch(e);ia=list(sa);ib=list(sb);assert ia==ib
  hashes.append(hashlib.sha256(json.dumps([ordered_a[i] for i in ia]).encode()).hexdigest())
 assert torch.equal(before,torch.get_rng_state())
 parent_pre=js(P/'GPU_PREFLIGHT.json');current_pre=js(O/'GPU_PREFLIGHT.json')
 result=dict(status='PASS_NO_PROTOCOL_DEVIATION_FOUND',seed=SEED,time=time.time(),config_differences=diffs,training_functions_AST_same=same,critical_source_hashes=source,dataset_order_and_payload_equal=True,dataset_size=1562,epoch_order_sha256=hashes,order_evidence_scope='Reconstructed from actual frozen loader, actual payloads, seed and sampler; historical per-batch trace not available',historical_gpu=parent_pre.get('gpu'),current_gpu=current_pre.get('gpu'),torch_current=torch.__version__,numeric_execution_limit='No guarantee of historical per-kernel/cuDNN algorithm identity; hardware is disclosed, not a proven sole cause',frozen_verify='PASS',optimizer_calls=0,environment_steps=0)
 init=O/'ra_coreset/INITIAL_STATE.json'
 if init.exists():
  result['initial_model_and_all_rng_equal']=js(init)==js(P/'coreset/INITIAL_STATE.json');assert result['initial_model_and_all_rng_equal']
  ea=js(O/'ra_coreset/epochs.json');eb=[x for x in js(P/'coreset/epochs.json') if x['task']==6]
  assert all(all(x[k]==y[k] for k in ['epoch','batches','steps','lr_first','lr_after']) for x,y in zip(ea,eb))
  result['task6_completed_epoch_schedule_and_batch_counts_equal']=True
  cp=O/'ra_coreset/recovery/latest.pth';r=torch.load(cp,map_location='cpu')
  assert r['contract_sha256']==sha(O/'CONTRACT.json') and r['modal_sha256']==sha(O/'MODAL_CONTRACT.json')
  assert r['task']==6 and r['next_epoch']==99 and not r['stage_complete'] and r['state']['step']==15543
  assert not r['buffer_hashes'] and not list((O/'ra_coreset/buffer').glob('*.pt'))
  assert set(r['state'])>={'model','optimizer','scheduler','grads','modes','rng','step'}
  result['resume_authority']=dict(checkpoint_sha256=sha(cp),task=6,next_epoch=99,committed_steps=15543,repeated_uncommitted_epoch99_steps=157,full_state_present=True)
 dest=N/'stage1'/str(SEED)/'PROTOCOL_EQUIVALENCE.json';assert not dest.exists();dump(dest,result)
 print(json.dumps(dict(status=result['status'],seed=SEED,dataset=True,AST=True,historical_gpu=result['historical_gpu'],current_gpu=result['current_gpu'])))
if __name__=='__main__':main()
