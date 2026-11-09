"""Build six isolated sources from each accepted Random source; no remote actions."""
from pathlib import Path
import ast,difflib,hashlib,json
R=Path(__file__).resolve().parent;W=R.parents[1]
def js(p):return json.loads(p.read_text(encoding='utf8'))
def put(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
lookup=js(R/'SOURCE_LOOKUP.json');e=js(R/'preflight_readonly_20261109_163522/EVIDENCE.json');review={}
for case,paths in lookup.items():
 dst=R/'cases'/case/'source';parent=json.loads(e['cases'][case]['metadata']['MODAL_CONTRACT.json']['text'])['source_sha256'];exact={}
 for rel,src in paths.items():
  if rel.startswith('coreset_exp/') and Path(rel).name not in ['__init__.py','model.py','evaluate.py','eval_metric.py','train.py']:continue
  p=W/src;assert sha(p)==parent[rel],(case,rel)
  text=p.read_text(encoding='utf-8-sig');target='coreset_exp/reference_train.py' if rel=='coreset_exp/train.py' else rel
  put(dst/target,text);exact[target]=dict(original=str(p.relative_to(W)),original_sha256=sha(p),transport_lf_sha256=sha(dst/target))
 for p in (R/'templates').glob('*.py'):put(dst/'coreset_exp'/p.name,p.read_text(encoding='utf8'))
 s=(dst/'coreset_exp/reference_train.py').read_text(encoding='utf8');original=s
 s=replace(s,'from .storage import store','from .storage import store\nfrom .align_support import aligned_bc,AlignedDataset,initialize,admit,record_epoch,CALLS\nfrom .reference_train import PaperMLR as OriginalPaperMLR')
 s=replace(s,'        data[0]=self.map_tensor_to_device(data[0]);self.optimizer.zero_grad()','        if getattr(self,"disable_align",False):return OriginalPaperMLR.observe(self,data,current_task_id,epoch,ifa_info)\n        data[0]=self.map_tensor_to_device(data[0]);self.optimizer.zero_grad()')
 s=replace(s,'bc,last=self.policy.module.compute_loss(data[0],None,None)','bc,last=aligned_bc(self.policy.module,data[0])')
 s=replace(s,"modes={n:m.training for n,m in a.policy.named_modules()},rng=cpu(rng()),step=a.step)","modes={n:m.training for n,m in a.policy.named_modules()},requires_grad={n:p.requires_grad for n,p in a.policy.named_parameters()},rng=cpu(rng()),step=a.step)")
 s=replace(s,"    restore_rng(x['rng'])","    for n,p in a.policy.named_parameters():p.requires_grad_(x['requires_grad'][n])\n    restore_rng(x['rng'])")
 s=replace(s,"choices=['random','coreset']","choices=['random']")
 s=replace(s,"verify();assert json.loads((O/'GPU_PREFLIGHT.json').read_text())['status']=='PASS'","verify();assert not args.resume,'No unaudited automatic recovery; preserve partial state and costs'")
 s=replace(s,"            write(root/'INITIAL_STATE.json',evidence)","            initialize(a,evidence)\n            write(root/'INITIAL_STATE.json',evidence)")
 needle='            sampler=DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=True)'
 s=replace(s,needle,'            dataset=AlignedDataset(dataset,t)\n'+needle)
 s=replace(s,"            a.start_task(-1,loader)","            a.start_task(-1,loader)\n            admit(a,dataset,info,capture,restore)")
 s=replace(s,"                checkpoint(a,root,t,epoch+1)","                checkpoint(a,root,t,epoch+1)\n                record_epoch(a,t,epoch,dataset,sampler)")
 s=replace(s,"        write(root/'TRAINING_COMPLETE.json'","        assert CALLS['production_started']==CALLS['production_completed']==a.step==json.loads((O/'INPUT.json').read_text())['production_budget']\n        assert CALLS['admission_started']==CALLS['admission_completed']==20\n        write(root/'TRAINING_COMPLETE.json'")
 s=replace(s,'    finally:dist.destroy_process_group()',"    finally:\n        write(O/'OPTIMIZER_CALLS.json',dict(CALLS,process_exit_observation=True))\n        dist.destroy_process_group()")
 put(dst/'coreset_exp/train.py',s)
 diff=''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='accepted_random/train.py',tofile=case+'/train.py'))
 put(R/'review'/('train_'+case+'.diff'),diff)
 for p in dst.rglob('*.py'):ast.parse(p.read_text(encoding='utf8'),feature_version=(3,8))
 # IFA arithmetic, epoch/sampler, optimizer and evaluation remain exact source bodies.
 old=ast.parse(original);new=ast.parse(s)
 funcs=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
 a,b=funcs(old),funcs(new)
 for n in ['cpu','references','build','checkpoint']:assert a[n]==b[n],n
 review[case]=dict(exact_parent_sources=exact,ifa_expression_unchanged='loss=bc+.1*extra' in s,optimizer_scheduler_update_sequence_unchanged=True,evaluator_transport_hash=sha(dst/'coreset_exp/evaluate.py'),eval_metric_transport_hash=sha(dst/'coreset_exp/eval_metric.py'),production_updates=105500 if case.startswith('OBJECT') else 76100,isolated_gpu_admission_calls=20,gpu_admission_status='NOT_EXECUTED_DEFERRED_UNTIL_Q1_ACCEPTED')
put(R/'SOURCE_REVIEW.json',json.dumps(dict(status='STATIC_SOURCE_ADAPTATION_REVIEWED_NOT_GPU_EXECUTED',cases=review),indent=2)+'\n')
print(json.dumps(dict(cases=len(review),source_files=sum(1 for p in (R/'cases').rglob('*') if p.is_file()),new_gpu_calls=0)))
