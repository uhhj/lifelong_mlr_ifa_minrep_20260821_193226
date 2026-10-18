"""CPU-only index reconstruction from frozen loader source; no payload/model/optimizer/environment."""
from pathlib import Path
import ast,os,random,json,hashlib,collections,time
import torch
torch.set_num_threads(2)
P=Path('/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed42_20261017')
S=Path('/root/mlr_ifa/experiments/SCv2_Coreset_GOAL_seed42_20261018')
SOURCE=Path('/root/mlr_ifa/experiment_workspaces/SCv2_Coreset_GOAL_seed42_20261018/source/libero/lifelong/algos/mlr.py')
OUT=Path('/tmp/scv2_batch_order_20261018.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p):return json.loads(p.read_text())
paths=[SOURCE]+[r/a/'storage'/f'task{t}.json' for r,a in [(P,'coreset'),(S,'sc_coreset')] for t in range(6,10)]
before={str(p):sha(p) for p in paths};start=time.time()
fn=next(x for x in ast.parse(SOURCE.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='build_dataset_for_task')
code=compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec')
def make(stage,root,arm):
 current=sorted((root/'features'/str(stage)).glob('*.pt'))
 # Reconstruct the PRE-stage buffer, not the final buffer containing future tasks.
 old=[]
 for t in range(6,stage):old.extend(root/arm/'buffer'/r['destination'] for r in js(root/arm/'storage'/f'task{t}.json')['selected'])
 old=sorted(old)
 def listing(d):
  if d==str(root/'features'/str(stage)):return [str(p) for p in current]
  assert d==str(root/arm/'buffer');return [str(p) for p in old]
 ns=dict(os=os,random=random,_list_pt=listing,FileTrajectoryDataset=lambda x:list(x))
 exec(code,ns);result=ns['build_dataset_for_task'](str(root/'features'),stage,str(root/arm/'buffer'),42,start=6)
 return [f't{stage}__{Path(p).name}' if Path(p).parent.name==str(stage) else Path(p).name for p in result]
out=[]
for t in range(6,10):
 a=make(t,P,'coreset');b=make(t,S,'sc_coreset');assert len(a)==len(b)==len(set(a))==len(set(b))
 n=len(a);common=set(a)&set(b);ai={v:i for i,v in enumerate(a)};bi={v:i for i,v in enumerate(b)}
 moved=[v for v in common if ai[v]!=bi[v]]
 current={v for v in common if v.startswith(f't{t}__')};assert all(ai[v]==bi[v] for v in current)
 assert all(x.split('__')[0]==y.split('__')[0] for x,y in zip(a,b))
 records=[]
 for epoch in range(100):
  # Actual torch 1.13 DistributedSampler indices, no DataLoader iteration or __getitem__.
  sampler=torch.utils.data.DistributedSampler(range(n),num_replicas=1,rank=0,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(epoch)
  order=list(sampler);assert len(order)==n and len(set(order))==n
  idx={i:j for j,i in enumerate(order)}
  changed_batches=0;current_touched=0;changed_slots=0;batch_set_changed=0
  for st in range(0,n,10):
   indices=order[st:st+10];x=[a[i] for i in indices];y=[b[i] for i in indices]
   if x!=y:
    changed_batches+=1;current_touched+=sum(v in current for v in x)
   if set(x)!=set(y):batch_set_changed+=1
   changed_slots+=sum(u!=v for u,v in zip(x,y))
  records.append(dict(epoch=epoch,batches=(n+9)//10,changed_ordered_batches=changed_batches,changed_batch_sets=batch_set_changed,
   changed_slots=changed_slots,current_samples_in_changed_batches=current_touched,
   common_samples_changed_update_step=sum(idx[ai[v]]//10!=idx[bi[v]]//10 for v in common)))
 out.append(dict(task=t,n=n,current_n=len(current),old_n=n-len(current),replaced_members=len(set(b)-set(a)),
  aligned_slots_changed=sum(x!=y for x,y in zip(a,b)),common_samples_moved=len(moved),current_positions_identical=True,task_slots_identical=True,
  coreset_order_sha256=hashlib.sha256(json.dumps(a).encode()).hexdigest(),scv2_order_sha256=hashlib.sha256(json.dumps(b).encode()).hexdigest(),
  epochs=records,summary={k:sum(x[k] for x in records) for k in records[0] if k!='epoch'}))
assert before=={str(p):sha(p) for p in paths}
result=dict(status='PASS',torch=torch.__version__,cpu_only=True,model_forwards=0,optimizer_updates=0,environment_calls=0,
 input_sha256=before,files_unchanged=True,stages=out,seconds=time.time()-start,
 interpretation='Deterministic reconstruction from exact frozen build_dataset_for_task AST and torch DistributedSampler; no historical per-batch log exists. No data tensors loaded.')
OUT.write_text(json.dumps(result,indent=2)+'\n');print(OUT)
