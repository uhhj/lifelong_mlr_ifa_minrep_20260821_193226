"""Isolated CPU BC probe on complete historical AdamW anchors. No rollout/training queue."""
import argparse,ast,contextlib,copy,gc,importlib.util,io,math,random,time,types
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
import robomimic.utils.tensor_utils as TensorUtils
from .common import *

def native_method(path,cls,name,scope):
    tree=ast.parse(path.read_text());body=tree.body if cls is None else next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    node=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope);return scope[name]

def filemodule(path,name):
    sp=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);return mod

def digest(x):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for r in v:visit(r)
        else:h.update(repr(v).encode())
    visit(x);return h.hexdigest()

def build(repo,s):
    tm=filemodule(repo/'libero/lifelong/models/modules/transformer_modules.py','r4_native_transformer')
    head=filemodule(repo/'libero/lifelong/models/policy_head.py','r4_native_head')
    scope=dict(torch=torch,nn=nn,TensorUtils=TensorUtils,DDP=torch.nn.parallel.DistributedDataParallel)
    brief=repo/'libero/lifelong/models/brief.py';base=repo/'libero/lifelong/models/base_policy.py';algo_src=repo/'libero/lifelong/algos/base.py'
    class Policy(nn.Module):
        def __init__(self):
            super().__init__();self.temporal_transformer=tm.OfficialGPT2Decoder(512,6,8,dropout=.15,max_position_embeddings=32,train=True);self.policy_head=head.GMMHead(512,7,num_modes=s['config']['policy']['num_gmm_modes'],min_std=1e-4)
    for method in ['temporal_encode','forward']:setattr(Policy,method,native_method(brief,'BriefPolicy',method,scope))
    Policy.compute_loss=native_method(base,'BasePolicy','compute_loss',scope)
    class Wrap(nn.Module):
        def __init__(self):super().__init__();self.module=Policy()
        def compute_loss(self,*args,**kwargs):return self.module.compute_loss(*args,**kwargs)
    a=types.SimpleNamespace();a.policy=Wrap();a.cfg=EasyDict(copy.deepcopy(s['config']));a.cfg.ifa_enabled=False;a.loss_scale=a.cfg.train.loss_scale;assert a.loss_scale==1
    a.optimizer=torch.optim.AdamW(a.policy.parameters(),**a.cfg.train.optimizer.kwargs)
    schedule=native_method(repo/'libero/lifelong/utils.py',None,'make_linear_schedule_with_warmup',scope)
    plan=read(repo/s['plan_path']);total=a.cfg.train.n_epochs*plan['batches_per_epoch'];warm=a.cfg.train.scheduler.kwargs.warmup_steps
    a.scheduler=torch.optim.lr_scheduler.LambdaLR(a.optimizer,schedule(warm,total))
    a.optimizer_step_and_maybe_schedule=types.MethodType(native_method(algo_src,'Sequential','optimizer_step_and_maybe_schedule',scope),a)
    a.observe=types.MethodType(native_method(algo_src,'Sequential','observe',scope),a)
    a.repro_scheduler_mode=a.cfg.repro_scheduler_mode;a.map_tensor_to_device=lambda x:x
    return a

def restore(a,s):
    a.policy.load_state_dict(s['model'],strict=True);a.optimizer.load_state_dict(copy.deepcopy(s['optimizer']));a.scheduler.load_state_dict(copy.deepcopy(s['scheduler']))
    a._optimizer_step_count=s['optimizer_step'];a._scheduler_step_count=s['scheduler_step']
    random.setstate(s['rng']['python']);np.random.set_state(s['rng']['numpy']);torch.set_rng_state(s['rng']['cpu'])
    for n,m in a.policy.named_modules():m.training=s['training_modes'][n]

def eval_mode(a):
    a.policy.eval();a.policy.module.training=True # native dispatch flag only; all dropout children stay eval

def checks(a,s):
    return dict(model=digest(a.policy.state_dict())==digest(s['model']),optimizer=digest(a.optimizer.state_dict())==digest(s['optimizer']),scheduler=digest(a.scheduler.state_dict())==digest(s['scheduler']),
      counters=a._optimizer_step_count==s['optimizer_step'] and a._scheduler_step_count==s['scheduler_step'],cpu_rng=torch.equal(torch.get_rng_state(),s['rng']['cpu']),numpy_rng=digest(np.random.get_state())==digest(s['rng']['numpy']),python_rng=random.getstate()==s['rng']['python'],modes={n:m.training for n,m in a.policy.named_modules()}==s['training_modes'])

def loss(a,z):return a.policy.compute_loss({k:v.unsqueeze(0) for k,v in z.items()},None,None)[0]
def grad(a,z):
    a.optimizer.zero_grad(set_to_none=True);v=loss(a,z);gg=torch.autograd.grad(v,tuple(a.policy.parameters()),allow_unused=True)
    return float(v),[None if g is None else g.detach() for g in gg]
def dot(g,h):return sum(float((x.double()*y.double()).sum()) for x,y in zip(g,h) if x is not None and y is not None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();repo,out=args.repo,args.out
    torch.set_num_threads(1);assert not torch.cuda.is_initialized();torch.use_deterministic_algorithms(True)
    source_files=['libero/lifelong/models/modules/transformer_modules.py','libero/lifelong/models/policy_head.py','libero/lifelong/models/brief.py','libero/lifelong/models/base_policy.py','libero/lifelong/algos/base.py','libero/lifelong/utils.py']
    write(out/'R4_RUNTIME_PROVENANCE.json',dict(files={p:sha(repo/p) for p in source_files},adapter=sha(__file__),mechanism='native classes loaded by exact file; native methods extracted unchanged by AST to avoid broken outer LIBERO package resolution; strict full state load plus independent native observe parity',device='CPU',torch=torch.__version__),True)
    labels={r['pair_id']:r for r in csv.DictReader((out/'semantic_audit/AI_SEMANTIC_LABELS.csv').open())};pairs=read(out/'AUDIT_SELECTION_PRIVATE.json')['pairs']
    for r in pairs:assert sha(out/'semantic_audit'/(r['pair_id']+'.png'))==labels[r['pair_id']]['image_sha256']
    starts=read(repo/'reports/actual_update_fidelity_v1/START_MANIFEST.json')['starts'];dm=read(repo/'reports/abcr_object_pilots_v1/DATA_MANIFEST.json')['demos'];actions={}
    for r in dm:
        if r['task']>=6:
            z=np.load(repo/'reports/abcr_object_pilots_v1'/(r['id']+'_raw.npz'));actions[(r['task'],r['demo'])]=z['actions']
    manifests=[];outputs=[];audit=[];allparity=[]
    for task in [7,8]:
        start=next(x for x in starts if x['seed']==45 and x['task_id']==task and x['anchor']['stage_step']==x['stage_step'])
        p=Path(start['anchor']['anchor']['path']);assert sha(p)==start['anchor']['anchor']['sha256'];plan=read(repo/start['plan']['path']);assert sha(repo/start['plan']['path'])==start['plan']['sha256']
        candidates=[r for r in pairs if r['task']==task and r['physical_pass'] and labels[r['pair_id']]['label']=='COMPARABLE']
        wanted={};resolved={}
        for r in candidates:
            for side,t in [('new',task),('old',r['old_task'])]:
                act=actions[(t,r[side+'_demo'])];w=r[side+'_window'];key=(t,np.ascontiguousarray(act[np.minimum(np.arange(w,w+8),len(act)-1)],dtype=np.float32).tobytes());wanted[(r['pair_id'],side)]=key
        needed=set(wanted.values())
        for origin,expected in zip(plan['immutable_feature_origins'],plan['record_sha256']):
            t=int(Path(origin).parent.name);z=torch.load(repo/origin,map_location='cpu');key=(t,np.ascontiguousarray(z['actions'].numpy().reshape(8,7),dtype=np.float32).tobytes())
            if key in needed:
                assert sha(repo/origin)==expected;resolved.setdefault(key,[]).append(dict(origin=origin,sha256=expected))
        chosen=[]
        for r in candidates:
            count={side:len(resolved.get(wanted[(r['pair_id'],side)],[])) for side in ['new','old']}
            if all(n==1 for n in count.values()):chosen.append(dict(r,inputs={side:resolved[wanted[(r['pair_id'],side)]][0] for side in ['new','old']}))
            else:audit.append(dict(pair_id=r['pair_id'],task=task,reason='NOT_UNIQUELY_IN_HISTORICAL_NATIVE_PLAN',matches=count))
        manifest=dict(task=task,start_id=start['start_id'],anchor=str(p),anchor_sha=sha(p),plan=start['plan'],selected=chosen,ai_physical_eligible=len(candidates),optimizer_state='full original moments restored; no initialization from model-only file')
        write(out/('R4_TASK%d_MANIFEST.json'%task),manifest,True);manifests.append(manifest)
        if not chosen:continue
        s=torch.load(p,map_location='cpu');a=build(repo,s);restore(a,s);chk=checks(a,s);assert all(chk.values());audit.append(dict(task=task,restore=chk,cuda_rng='preserved in immutable source but not executed on CPU; no cross-device bitwise trajectory claim'))
        for index,r in enumerate(chosen):
            dest=out/('R4_%s.json'%r['pair_id']);binding=dict(protocol=sha(out/'PROTOCOL.json'),manifest=sha(out/('R4_TASK%d_MANIFEST.json'%task)),source=sha(__file__),labels=sha(out/'semantic_audit/AI_SEMANTIC_LABELS.csv'))
            if dest.exists():
                prior=read(dest);assert prior['binding']==binding;outputs.append(prior['result']);continue
            begin=time.time();z={side:torch.load(repo/spec['origin'],map_location='cpu') for side,spec in r['inputs'].items()}
            restore(a,s);eval_mode(a);old0,go=grad(a,z['old']);new0,gn=grad(a,z['new']);cos=dot(go,gn)/math.sqrt(dot(go,go)*dot(gn,gn))
            a.optimizer.zero_grad(set_to_none=True)
            for par,g in zip(a.policy.parameters(),gn):par.grad=None if g is None else g.clone()
            clip=float(nn.utils.clip_grad_norm_(a.policy.parameters(),a.cfg.train.grad_clip));a.optimizer_step_and_maybe_schedule()
            hd=0.;dn=0.
            for (name,par),g in zip(a.policy.named_parameters(),go):
                delta=par.detach().double()-s['model'][name].double();dn+=float(delta.square().sum())
                if g is not None:hd+=float((g.double()*delta).sum())
            with torch.no_grad():old1=float(loss(a,z['old']));new1=float(loss(a,z['new']))
            counts=dict(optimizer_delta=a._optimizer_step_count-s['optimizer_step'],scheduler_delta=a._scheduler_step_count-s['scheduler_step']);assert counts==dict(optimizer_delta=1,scheduler_delta=1)
            if index==0:
                proposal={k:digest(v) for k,v in [('model',a.policy.state_dict()),('optimizer',a.optimizer.state_dict()),('scheduler',a.scheduler.state_dict())]}
                restore(a,s);eval_mode(a)
                data=[{k:v.unsqueeze(0) for k,v in z['new'].items()},[str(task)]]
                with contextlib.redirect_stdout(io.StringIO()):a.observe(data,task,start['epoch'],None)
                native={k:digest(v) for k,v in [('model',a.policy.state_dict()),('optimizer',a.optimizer.state_dict()),('scheduler',a.scheduler.state_dict())]}
                parity={k:proposal[k]==native[k] for k in proposal};assert all(parity.values()),parity;allparity.append(dict(task=task,pair_id=r['pair_id'],checks=parity));write(out/('R4_TASK%d_PARITY.json'%task),allparity[-1],True)
            restore(a,s);restored=checks(a,s);assert all(restored.values())
            tau=32*np.finfo(np.float32).eps*max(1,abs(old0));outcome='HARM' if old1-old0>tau else ('BENEFIT' if old1-old0 < -tau else 'NEUTRAL')
            result=dict(pair_id=r['pair_id'],task=task,old_task=r['old_task'],action_label=r['label'],new_demo=r['new_demo'],old_demo=r['old_demo'],start_id=start['start_id'],optimizer_step=s['optimizer_step'],lr=s['optimizer']['param_groups'][0]['lr'],gradient_cosine=cos,gradient_angle_degrees=math.degrees(math.acos(max(-1,min(1,cos)))),old_base_loss=old0,new_base_loss=new0,old_delta=old1-old0,new_delta=new1-new0,old_gradient_dot_actual_update=hd,actual_update_l2=math.sqrt(dn),preclip_new_norm=clip,deadband=tau,old_outcome=outcome,seconds=time.time()-begin,**counts)
            write(dest,dict(binding=binding,result=result,restored=restored));outputs.append(result);print('R4',r['pair_id'],r['label'],outcome,'seconds',result['seconds'],flush=True)
            del go,gn,z;gc.collect()
        del a,s;gc.collect()
    table(out/'R4_RESULTS.csv',outputs);write(out/'R4_RESTORE_AND_PARITY.json',dict(audit=audit,parity=allparity,formal_training_resumed=False))
    write(out/'R4_COMPLETE.json',dict(status='MEASURED' if outputs else 'INSUFFICIENT_VALID_PAIRS',n=len(outputs),tasks_measured=sorted(set(r['task'] for r in outputs)),task9='NOT_RUN_NO_COMPLETE_OPTIMIZER_STATE',formal_training_resumed=False,cuda_initialized=torch.cuda.is_initialized()))

if __name__=='__main__':main()
