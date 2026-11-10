import argparse,copy,json,os,time,csv,hashlib,pickle
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from .common import O,write,save,sha,verify,rng,restore_rng,hashes,jsonable
from .model import cfg,angular_mean,strict_downstream
from .storage import store
from .align_support import aligned_bc,AlignedDataset,initialize,admit,record_epoch,CALLS
from .reference_train import PaperMLR as OriginalPaperMLR
from libero.lifelong.algos.mlr import MLR,build_dataset_for_task,build_ifa_similarity_matrix
from libero.lifelong.utils import control_seed,get_task_embs
from libero.libero.benchmark import get_benchmark

def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return copy.deepcopy(x)
def references(c,t):
    out={};maxdev={}
    for task in range(6,t+1):
        files=sorted((O/'features'/str(t)).glob('*.pt')) if task==t else sorted(Path(c.buffer_dir).glob(f't{task}__*.pt'))
        assert files;ref=None;dev=0.
        for p in files:
            v=torch.load(p,map_location='cpu')['feature'][0,:,0,:]
            if ref is None:ref=v[0].clone()
            dev=max(dev,float((v-ref).abs().max()))
        assert dev<1e-4,('Non-stable language references: protocol decision needed',task,dev)
        out[task]=ref.cuda();maxdev[task]=dev
    return out,maxdev
class AlignPaperMLR(MLR):
    def observe(self,data,current_task_id,epoch,ifa_info=None):
        if getattr(self,"disable_align",False):return OriginalPaperMLR.observe(self,data,current_task_id,epoch,ifa_info)
        data[0]=self.map_tensor_to_device(data[0]);self.optimizer.zero_grad()
        bc,last=aligned_bc(self.policy.module,data[0])
        mask=torch.tensor([int(x)==current_task_id for x in data[1]],device=last.device)
        pairs=ifa_info['selected_pairs'];olds=[self.refs[j if k==current_task_id else k] for j,k in pairs]
        extra=angular_mean(last[mask],self.refs[current_task_id],olds,self.cfg.gamma) if mask.any() else last.sum()*0
        loss=bc+.1*extra;assert torch.isfinite(loss)
        (self.loss_scale*loss).backward()
        if self.cfg.train.grad_clip is not None:torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.cfg.train.grad_clip)
        self.optimizer.step();self.scheduler.step()
        self.step+=1;return float(loss.detach())
def build(c):
    control_seed(42);b=get_benchmark('LIBERO_OBJECT')(0)
    b.set_task_embs(get_task_embs(c,[b.get_task(i).language for i in range(10)]))
    Path(c.experiment_dir).mkdir(parents=True,exist_ok=True);Path(c.buffer_dir).mkdir(parents=True,exist_ok=True)
    a=AlignPaperMLR(1,c).cuda();n=strict_downstream(a.policy,c.pretrain)
    assert all(n.startswith(('temporal_transformer.','policy_head.')) for n,p in a.policy.named_parameters() if p.requires_grad)
    a.policy=DDP(a.policy,device_ids=[0],output_device=0);a.train();a.step=0
    return a,b
def capture(a):
    return dict(model=cpu(a.policy.state_dict()),optimizer=cpu(a.optimizer.state_dict()),scheduler=cpu(a.scheduler.state_dict()),
        grads={n:cpu(p.grad) if p.grad is not None else None for n,p in a.policy.named_parameters()},
        modes={n:m.training for n,m in a.policy.named_modules()},requires_grad={n:p.requires_grad for n,p in a.policy.named_parameters()},rng=cpu(rng()),step=a.step)
def restore(a,x):
    a.policy.load_state_dict(x['model'],strict=True);a.optimizer.load_state_dict(x['optimizer']);a.scheduler.load_state_dict(x['scheduler']);a.step=x['step']
    for n,p in a.policy.named_parameters():p.grad=None if x['grads'][n] is None else x['grads'][n].to(p.device)
    for n,m in a.policy.named_modules():m.training=x['modes'][n]
    for n,p in a.policy.named_parameters():p.requires_grad_(x['requires_grad'][n])
    restore_rng(x['rng'])
def checkpoint(a,root,t,epoch,complete=False):
    save(root/'recovery/latest.pth',dict(state=capture(a),task=t,next_epoch=epoch,stage_complete=complete,
        contract_sha256=sha(O/'CONTRACT.json'),modal_sha256=sha(O/'MODAL_CONTRACT.json'),buffer_hashes=hashes(Path(a.cfg.buffer_dir))))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['random']);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    verify();assert not args.resume,'No unaudited automatic recovery; preserve partial state and costs'
    assert (O/'FEATURES.json').is_file();c=cfg(args.arm);root=Path(c.run_root)
    assert not (root/'TRAINING_COMPLETE.json').exists(),'Already complete'
    cp=root/'recovery/latest.pth';assert args.resume or not cp.exists(),'Explicit resume required'
    resumed=torch.load(cp,map_location='cpu') if args.resume else None
    dist.init_process_group('nccl');torch.cuda.set_device(0);begin=time.time()
    try:
        a,b=build(c);history=[]
        if not resumed:
            r=rng()
            digest={k:hashlib.sha256(pickle.dumps(v) if k in ('python','numpy') else b''.join(x.cpu().numpy().tobytes() for x in (v if isinstance(v,list) else [v]))).hexdigest() for k,v in r.items()}
            model_digest={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in a.policy.state_dict().items()}
            evidence=dict(rng=digest,tensors=model_digest)
            initialize(a,evidence)
            write(root/'INITIAL_STATE.json',evidence)
            if args.arm=='coreset':assert evidence==json.loads((O/'random/INITIAL_STATE.json').read_text()),'Initialization differs'
        if resumed:
            assert resumed['contract_sha256']==sha(O/'CONTRACT.json') and resumed['modal_sha256']==sha(O/'MODAL_CONTRACT.json')
            a.policy.load_state_dict(resumed['state']['model'],strict=True);a.step=resumed['state']['step'];restore_rng(resumed['state']['rng'])
            actual=hashes(Path(c.buffer_dir));expected=resumed['buffer_hashes']
            assert all(actual.get(n)==h for n,h in expected.items())
            if actual!=expected:assert resumed['next_epoch']==100 and not resumed['stage_complete']
            history=json.loads((root/'epochs.json').read_text())
            # Recovery cp is commit authority; preserve uncommitted log tail as evidence.
            keep=[x for x in history if x['task']<resumed['task'] or x['task']==resumed['task'] and x['epoch']<resumed['next_epoch']]
            if history!=keep:write(root/f'interrupted_epochs_{time.time_ns()}.json',history);history=keep
        else:assert not list(Path(c.buffer_dir).glob('*.pt'))
        for t in range(6,10):
            if resumed and (t<resumed['task'] or t==resumed['task'] and resumed['stage_complete']):continue
            start=resumed['next_epoch'] if resumed and t==resumed['task'] else 0
            if start==100:
                store(c,t,b);resumed['stage_complete']=True;resumed['buffer_hashes']=hashes(Path(c.buffer_dir));save(cp,resumed);continue
            info=build_ifa_similarity_matrix(t,c.feature_dir,c.buffer_dir);a.refs,dev=references(c,t)
            write(root/f'ifa_task{t}.json',dict(info=jsonable(info),reference_max_deviation=dev))
            dataset=build_dataset_for_task(c.feature_dir,t,c.buffer_dir,42,start=6)
            dataset=AlignedDataset(dataset,t)
            sampler=DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=True)
            loader=DataLoader(dataset,batch_size=10,num_workers=c.train.num_workers,sampler=sampler,persistent_workers=False)
            a.start_task(-1,loader)
            admit(a,dataset,info,capture,restore)
            if resumed and t==resumed['task']:restore(a,resumed['state'])
            for epoch in range(start,100):
                sampler.set_epoch(epoch);a.policy.train();loss=0.;tic=time.time();lr_first=a.optimizer.param_groups[0]['lr']
                for batch in loader:loss+=a.observe(batch,t,epoch,info)
                row=dict(task=t,epoch=epoch,batches=len(loader),steps=a.step,loss=loss/len(loader),
                    lr_first=lr_first,lr_after=a.optimizer.param_groups[0]['lr'],seconds=time.time()-tic)
                history.append(row);write(root/'epochs.json',history)
                if epoch==99:
                    final=root/'models'/f'mlrifa_{t}_model_ep99.pth'
                    save(final,dict(state_dict=cpu(a.policy.state_dict()),cfg=c,task=t,epoch_index=99,
                        modal_sha256=sha(O/'MODAL_CONTRACT.json'),contract_sha256=sha(O/'CONTRACT.json')))
                    if t==6 and args.arm=='coreset':
                        old=torch.load(O/'random/models/mlrifa_6_model_ep99.pth',map_location='cpu')['state_dict'];now=cpu(a.policy.state_dict())
                        diff={k:float((now[k].double()-old[k].double()).abs().max()) for k in now}
                        same=set(old)==set(now) and all(v==0 for v in diff.values())
                        write(O/'TASK6_TENSOR_PARITY.json',dict(equal=same,max_abs_diff=diff,compared='actual state_dict tensors, no file metadata'))
                        assert same,'Task6 models differ before selection; stop C before task7'
                checkpoint(a,root,t,epoch+1)
                record_epoch(a,t,epoch,dataset,sampler)
                write(O/'RUN_STATUS.json',dict(status='TRAINING',arm=args.arm,task=t,completed_epoch=epoch,optimizer_steps=a.step,updated_unix=time.time(),pid=os.getpid()))
                print(json.dumps(row),flush=True)
            store(c,t,b);checkpoint(a,root,t,100,True);resumed=None
        assert len(history)==400 and all(sum(r['task']==t for r in history)==100 for t in range(6,10))
        assert CALLS['production_started']==CALLS['production_completed']==a.step==json.loads((O/'INPUT.json').read_text())['production_budget']
        assert CALLS['admission_started']==CALLS['admission_completed']==20
        write(root/'TRAINING_COMPLETE.json',dict(epochs=400,steps=a.step,seconds_this_invocation=time.time()-begin,
            summed_epoch_seconds=sum(x['seconds'] for x in history),checkpoints=hashes(root/'models','*.pth')))
    finally:
        write(O/'OPTIMIZER_CALLS.json',dict(CALLS,process_exit_observation=True))
        dist.destroy_process_group()
if __name__=='__main__':main()
