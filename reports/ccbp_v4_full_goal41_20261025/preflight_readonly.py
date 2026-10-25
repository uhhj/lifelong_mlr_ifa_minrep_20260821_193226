CPU_SOURCE={'legacy_losses': '"""Frozen CCBP-v1 and two execution-matched controls; no trainable additions."""\nimport torch\nimport torch.nn.functional as F\nGROUPS=((0,3),(3,6),(6,7))\ndef cost(mu,sd,y,scale):\n return sum(((mu[:,:,a:b]-y[:,None,a:b]).square()+sd[:,:,a:b].square()).mean(-1)/scale[g].square() for g,(a,b) in enumerate(GROUPS))\ndef w2(mu,sd,tm,ts,scale):\n return sum(((mu[:,:,a:b]-tm[:,None,a:b]).square()+(sd[:,:,a:b]-ts[:,None,a:b]).square()).mean(-1)/scale[g].square() for g,(a,b) in enumerate(GROUPS))\ndef route(logits,good):\n good=good.detach();allgood=good.all(-1);assert good.any(-1).all()\n # Finite sentinels avoid inf*0 and retain an exact zero for all-good rows.\n hi=logits.masked_fill(~good,-1e9).max(-1).values\n bad=logits.masked_fill(good,-1e9).max(-1).values\n return torch.where(allgood,logits.sum(-1)*0,F.relu(.1+bad-hi))\ndef aux(arm,q,y,teacher,scale):\n mu,sd,logits=q;tm,ts,tl=teacher;ix=torch.arange(len(y),device=y.device)\n winner=logits.detach().argmax(-1);tk=tl.argmax(-1)\n cc=cost(mu,sd,y,scale);selected=cc[ix,winner]\n budget=cost(tm,ts,y,scale)[ix,tk].detach()+.05\n if arm==\'A\':return mu.sum()*0,dict(count=len(y))\n if arm==\'B\':\n  d=w2(mu,sd,tm[ix,tk],ts[ix,tk],scale);value=d[ix,winner];good=F.one_hot(d.detach().argmin(-1),5).bool();empty=torch.zeros(len(y),device=y.device,dtype=torch.bool)\n elif arm==\'C\':\n  value=selected;good=F.one_hot(cc.detach().argmin(-1),5).bool();empty=torch.zeros(len(y),device=y.device,dtype=torch.bool)\n else:\n  assert arm==\'D\';good=cc.detach()<=budget[:,None];empty=~good.any(-1)\n  value=F.relu(selected-budget)+empty.to(selected.dtype)*F.relu(cc.min(-1).values-budget)\n  good=good| (empty[:,None]&F.one_hot(cc.detach().argmin(-1),5).bool())\n rr=route(logits,good);loss=(value+.1*rr).mean()\n return loss,dict(count=len(y),expert_selected_sum=float(selected.detach().sum()),teacher_budget_sum=float(budget.sum()),excess_sum=float(F.relu(selected-budget).detach().sum()),route_sum=float(rr.detach().sum()),empty=int(empty.sum()),all_good=int(good.all(-1).sum()))\n', 'losses': '"""Unmodified C expert path plus separate executed-distribution retention."""\nimport torch\nimport torch.nn.functional as F\nfrom legacy_losses import cost,w2,route\n\ndef expert(q,y,scale):\n    mu,sd,z=q;ix=torch.arange(len(mu),device=mu.device);k=z.detach().argmax(-1)\n    c=cost(mu,sd,y.detach(),scale.detach())\n    rr=route(z,F.one_hot(c.detach().argmin(-1),z.shape[-1]).bool())\n    return (c[ix,k]+.1*rr).mean(),c[ix,k].mean(),rr.mean()\n\ndef retain(q,tm,ts,scale):\n    mu,sd,z=q;ix=torch.arange(len(mu),device=mu.device);k=z.detach().argmax(-1)\n    d=w2(mu,sd,tm.detach(),ts.detach(),scale.detach())\n    rr=route(z,F.one_hot(d.detach().argmin(-1),z.shape[-1]).bool())\n    return d[ix,k],rr\n', 'synthetic_checks': '"""Numerical checks for this candidate only; not real-model or environment acceptance."""\nimport torch\nfrom losses import expert,retain\nfrom legacy_losses import aux as legacy_aux\n\ndef run():\n    torch.manual_seed(41)\n    mu=torch.randn(6,5,7,requires_grad=True);sd=torch.full_like(mu,.2,requires_grad=True);z=torch.randn(6,5,requires_grad=True)\n    q=(mu,sd,z);target=torch.randn(6,7);scale=torch.tensor([.4,.1,1.]);teacher=tuple(x.detach().clone().requires_grad_(True) for x in q)\n    value,_,_=expert(q,target,scale);reference,_=legacy_aux(\'C\',q,target,tuple(x.detach() for x in q),scale)\n    assert torch.equal(value,reference)\n    g0=torch.autograd.grad(value,q,retain_graph=True);g1=torch.autograd.grad(reference,q,retain_graph=True)\n    assert all(torch.equal(a,b) for a,b in zip(g0,g1))\n    ix=torch.arange(6);tk=teacher[2].argmax(-1);tm=teacher[0][ix,tk];ts=teacher[1][ix,tk]\n    a,b=retain(q,tm,ts,scale);loss=(a+.1*b).mean();loss.backward()\n    assert all(x.grad is None for x in teacher)\n    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in q)\n    perm=torch.tensor([3,1,4,0,2]);aa,bb=retain((mu[:,perm],sd[:,perm],z[:,perm]),tm,ts,scale)\n    assert torch.allclose(a,aa,atol=1e-6,rtol=1e-6) and torch.allclose(b,bb,atol=1e-6,rtol=1e-6)\n    return dict(status=\'PASS_SYNTHETIC_ONLY\',C_loss_and_output_gradient_exact=True,teacher_detached=True,finite_student_gradients=True,component_permutation_invariance=True,model_loads=0,environment_episodes=0)\n\nif __name__==\'__main__\':\n    import json\n    print(json.dumps(run(),indent=2))\n'}
"""Read-only parent, init-state split and storage check. No environment/model forwards."""
import os,sys,json,time,hashlib,subprocess,contextlib,io
from pathlib import Path
import types
P=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V3_FULL_GOAL41_20261025_v1_engineering02')
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V4_FULL_GOAL41_20261025_v1')
sys.path.insert(0,str(P))
from runtime import verify,sha,js,O,torch,np,EasyDict
from libero.libero.benchmark import get_benchmark
assert verify()=='232a34dbef9b268f4be82580f0b4299a2ecb944f436c1aac8fc146088f41243b'
assert all((P/x).read_text().strip()=='0' for x in ['controller.exit','evaluate.exit','train_C.exit','train_R.exit','train_G.exit'])
assert not Q.exists(),'No redeploy'
c=EasyDict(js(O/'resolved_eval.json'))
with contextlib.redirect_stdout(io.StringIO()):
    bench=get_benchmark(c.benchmark_name)(c.data.task_order_index)
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
# All locally retained outcome ledgers: initial-state identity only, never trajectory payloads.
used=set();ledgers=[]
for f in Path('/data/mlr_ifa/experiment_workspaces').rglob('OUTCOMES.json'):
    records=js(f)
    if not isinstance(records,list):continue
    for row in records:
        if isinstance(row,dict) and isinstance(row.get('initial_sha256'),str):used.add(row['initial_sha256'])
    ledgers.append(dict(path=str(f),sha256=sha(f),records=len(records)))
split={}
for taskid in range(6,10):
    t=bench.get_task(taskid);f=Path(c.init_states_folder)/t.problem_folder/t.init_states_file
    x=torch.load(f,map_location='cpu');assert len(x)>=30
    training={str(i):arrsha(np.asarray(x[i])) for i in range(20,30)}
    evaluation={str(i):arrsha(np.asarray(x[i])) for i in range(20)}
    assert len(set(training.values()))==10 and not set(training.values())&set(evaluation.values())
    assert not set(training.values())&used,('Training state previously evaluated',taskid)
    split[str(taskid)]=dict(path=str(f),file_sha256=sha(f),available=len(x),training=training,evaluation=evaluation)
processes=[]
for p in Path('/proc').glob('[0-9]*'):
    try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except OSError:continue
    if int(p.name)!=os.getpid() and any(str(z) in cmd for z in [P,Q]):processes.append(dict(pid=int(p.name),command=cmd))
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
cap={}
for mount,bound,reserve,rootfree in [('/root',0,1024**3,False),('/data',4*1024**3,2*1024**3,True)]:
    s=os.statvfs(mount);free=(s.f_bfree if rootfree else s.f_bavail)*s.f_frsize
    cap[mount]=dict(free=free,new_peak_bound=bound,reserve=reserve,headroom=free-bound-reserve)
hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
review={}
for name,source in CPU_SOURCE.items():
    module=types.ModuleType(name);sys.modules[name]=module;exec(compile(source,name+'.py','exec'),module.__dict__);review[name]=hashlib.sha256(source.encode()).hexdigest()
synthetic=sys.modules['synthetic_checks'].run()
print(json.dumps(dict(synthetic=synthetic,reviewed_source_sha256=review,status='PASS_PREDEPLOY_READONLY' if not gpu and not processes and all(x['headroom']>=0 for x in cap.values()) else 'REVIEW_RESOURCE_REQUIRED',observed=time.time(),parent_contract=verify(),source_files=len(js(P/'CONTRACT.json')['files']),initial_split=split,existing_outcome_ledgers=ledgers,hardware=hardware,gpu=gpu,processes=processes,capacity=cap,remote_writes=0,model_forwards=0,optimizer_updates=0,environments=0)))
