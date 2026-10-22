from coreset_exp.common import SEED,SELECTOR_SEED
import argparse,copy,json,os,time,csv,hashlib,pickle
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from coreset_exp.common import O,write,save,sha,verify,rng,restore_rng,hashes,jsonable
from coreset_exp.model import cfg,angular_mean,strict_downstream
from ra_exp.storage import store
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
from caet_v2_exp.observe import CAETv2MLR
from caet_v2_exp.dataset import EffectDataset
from caet_v2_exp.variant import VARIANT
def build(c):
    control_seed(SEED);b=get_benchmark('LIBERO_OBJECT')(0)
    b.set_task_embs(get_task_embs(c,[b.get_task(i).language for i in range(10)]))
    Path(c.experiment_dir).mkdir(parents=True,exist_ok=True);Path(c.buffer_dir).mkdir(parents=True,exist_ok=True)
    a=CAETv2MLR(1,c).cuda();n=strict_downstream(a.policy,c.pretrain)
    assert all(n.startswith(('temporal_transformer.','policy_head.')) for n,p in a.policy.named_parameters() if p.requires_grad)
    a.policy=DDP(a.policy,device_ids=[0],output_device=0);a.train();a.step=0
    return a,b
def capture(a):
    return dict(model=cpu(a.policy.state_dict()),optimizer=cpu(a.optimizer.state_dict()),scheduler=cpu(a.scheduler.state_dict()),
        grads={n:cpu(p.grad) if p.grad is not None else None for n,p in a.policy.named_parameters()},
        modes={n:m.training for n,m in a.policy.named_modules()},rng=cpu(rng()),step=a.step,v2_state=cpu(a.export_v2_state()))
def restore(a,x):
    a.policy.load_state_dict(x['model'],strict=True);a.optimizer.load_state_dict(x['optimizer']);a.scheduler.load_state_dict(x['scheduler']);a.step=x['step']
    for n,p in a.policy.named_parameters():p.grad=None if x['grads'][n] is None else x['grads'][n].to(p.device)
    for n,m in a.policy.named_modules():m.training=x['modes'][n]
    a.import_v2_state(x['v2_state']);restore_rng(x['rng'])
def checkpoint(a,root,t,epoch,complete=False):
    save(root/'recovery/latest.pth',dict(state=capture(a),task=t,next_epoch=epoch,stage_complete=complete,
        contract_sha256=sha(O/'CONTRACT.json'),modal_sha256=sha(O/'MODAL_CONTRACT.json'),buffer_hashes=hashes(Path(a.cfg.buffer_dir))))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['ra_coreset']);ap.add_argument('--resume',action='store_true');ap.add_argument('--through-task',type=int,choices=[6,7,8,9],required=True);args=ap.parse_args()
    verify();assert json.loads((O/'GPU_PREFLIGHT.json').read_text())['status']=='PASS'
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
            write(root/'INITIAL_STATE.json',evidence)
            assert evidence==json.loads((Path(os.environ['RA_BASE'])/'ra_coreset/INITIAL_STATE.json').read_text()),'Initialization differs from parent'
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
        for t in range(6,args.through_task+1):
            if resumed and (t<resumed['task'] or t==resumed['task'] and resumed['stage_complete']):continue
            start=resumed['next_epoch'] if resumed and t==resumed['task'] else 0
            if start==100:
                store(c,t,b);resumed['stage_complete']=True;resumed['buffer_hashes']=hashes(Path(c.buffer_dir));save(cp,resumed);continue
            info=build_ifa_similarity_matrix(t,c.feature_dir,c.buffer_dir);a.refs,dev=references(c,t)
            write(root/f'ifa_task{t}.json',dict(info=jsonable(info),reference_max_deviation=dev))
            dataset=build_dataset_for_task(c.feature_dir,t,c.buffer_dir,SEED,start=6)
            dataset=EffectDataset(dataset,t)
            sampler=DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=True)
            loader=DataLoader(dataset,batch_size=10,num_workers=c.train.num_workers,sampler=sampler,persistent_workers=False)
            a.start_task(-1,loader)
            a.configure_effect(t,dataset)
            if resumed and t==resumed['task']:restore(a,resumed['state'])
            for epoch in range(start,100):
                sampler.set_epoch(epoch);a.policy.train();a.caet_epoch_statistics=[];loss=0.;tic=time.time();lr_first=a.optimizer.param_groups[0]['lr']
                for batch in loader:loss+=a.observe(batch,t,epoch,info)
                a.finish_epoch(t,epoch)
                row=dict(task=t,epoch=epoch,batches=len(loader),steps=a.step,loss=loss/len(loader),
                    lr_first=lr_first,lr_after=a.optimizer.param_groups[0]['lr'],seconds=time.time()-tic)
                history.append(row);write(root/'epochs.json',history)
                write(root/'auxiliary'/f'task{t}_epoch{epoch:02d}.json',a.caet_epoch_statistics)
                if epoch==99:
                    final=root/'models'/f'mlrifa_{t}_model_ep99.pth'
                    save(final,dict(state_dict=cpu(a.policy.state_dict()),cfg=c,task=t,epoch_index=99,
                        modal_sha256=sha(O/'MODAL_CONTRACT.json'),contract_sha256=sha(O/'CONTRACT.json')))
                    save(root/'aux_heads'/('task%d_ep99.pth'%t),dict(state=cpu(a.export_aux_state()),task=t,epoch=99,contract_sha256=sha(O/'CONTRACT.json')))
                    if t==6:
                        old=torch.load(Path(os.environ['RA_BASE'])/'ra_coreset/models/mlrifa_6_model_ep99.pth',map_location='cpu')['state_dict'];now=cpu(a.policy.state_dict())
                        diff={k:float((now[k].double()-old[k].double()).abs().max()) for k in now}
                        same=set(old)==set(now) and all(v==0 for v in diff.values())
                        write(O/'TASK6_TENSOR_PARITY.json',dict(equal=same,max_abs_diff=diff,compared='actual state_dict tensors, no file metadata'))
                        assert same,'Task6 models differ before selection; stop C before task7'
                checkpoint(a,root,t,epoch+1)
                write(O/'RUN_STATUS.json',dict(status='TRAINING',arm=args.arm,task=t,completed_epoch=epoch,optimizer_steps=a.step,updated_unix=time.time(),pid=os.getpid()))
                print(json.dumps(row),flush=True)
            store(c,t,b);checkpoint(a,root,t,100,True);resumed=None
        if args.through_task<9:
            write(root/f'TASK{args.through_task}_TRAINING_COMPLETE.json',dict(status='PASS',epochs=len(history),steps=a.step,through_task=args.through_task))
            return
        assert len(history)==400 and all(sum(r['task']==t for r in history)==100 for t in range(6,10))
        write(root/'TRAINING_COMPLETE.json',dict(epochs=400,steps=a.step,seconds_this_invocation=time.time()-begin,
            summed_epoch_seconds=sum(x['seconds'] for x in history),checkpoints=hashes(root/'models','*.pth')))
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
