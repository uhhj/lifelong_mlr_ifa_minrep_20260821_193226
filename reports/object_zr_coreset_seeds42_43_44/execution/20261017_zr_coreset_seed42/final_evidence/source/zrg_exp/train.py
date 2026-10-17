import argparse,json,time,copy,shutil,hashlib
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from .common import *
from .model import cfg,attach,register,strict_downstream
from .diagnostics import check
from coreset_exp.train import PaperMLR,capture,restore
from libero.lifelong.algos.mlr import build_dataset_for_task,build_ifa_similarity_matrix
from libero.lifelong.utils import control_seed,get_task_embs,make_linear_schedule_with_warmup
from libero.libero.benchmark import get_benchmark

class ZRMLR(PaperMLR):
    def start_task(self,task,loader):
        self.current_task=task
        params=[p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer=torch.optim.AdamW(params,**self.cfg.train.optimizer.kwargs)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,make_linear_schedule_with_warmup(500,100*len(loader)),last_epoch=-1)
def build(c):
    control_seed(42);b=get_benchmark('LIBERO_OBJECT')(0);b.set_task_embs(get_task_embs(c,[b.get_task(i).language for i in range(10)]))
    Path(c.experiment_dir).mkdir(parents=True,exist_ok=True);Path(c.buffer_dir).mkdir(parents=True,exist_ok=True)
    a=ZRMLR(1,c).cuda();strict_downstream(a.policy,c.pretrain);attach(a.policy);a.train();a.step=0
    return a,b
def stage(a,t):
    if isinstance(a.policy,DDP):
        p=a.policy.module;a.policy=p
    p=a.policy;event=register(p,t);r=rng_digest();a.policy=DDP(p,device_ids=[0],output_device=0)
    assert r==rng_digest();event['ddp_rng_unchanged']=True
    return event
def refs(c,t):
    out={};dev={}
    for j in range(6,t+1):
        paths=sorted((P/'features'/str(t)).glob('*.pt')) if j==t else sorted(Path(c.buffer_dir).glob(f't{j}__*.pt'))
        ref=None;d=0.
        for path in paths:
            v=torch.load(path,map_location='cpu')['feature'][0,:,0]
            if ref is None:ref=v[0].clone()
            d=max(d,float((v-ref).abs().max()))
        assert ref is not None and d<1e-4;out[j]=ref.cuda();dev[j]=d
    return out,dev
def store(c,t):
    start=time.time();orig=json.loads((P/'coreset/storage'/f'task{t}.json').read_text());f=ROOT/'storage'/f'task{t}.json'
    if f.exists():m=json.loads(f.read_text());assert m['contract_sha256']==sha(O/'CONTRACT.json')
    else:
        assert hashes(Path(c.buffer_dir))==orig['before']
        m=copy.deepcopy(orig);m['parent_manifest_sha256']=sha(P/'coreset/storage'/f'task{t}.json');m['contract_sha256']=sha(O/'CONTRACT.json');write(f,m)
    assert m['selected']==orig['selected'];allowed=dict(orig['before']);allowed.update({r['destination']:r['sha256'] for r in orig['selected']})
    actual=hashes(Path(c.buffer_dir));assert all(n in allowed and allowed[n]==h for n,h in actual.items())
    for r in m['selected']:
        src=Path(r['source']);dst=Path(c.buffer_dir)/r['destination'];assert sha(src)==r['sha256']
        if dst.exists():assert sha(dst)==r['sha256']
        else:
            tmp=dst.with_suffix('.copying');shutil.copy2(src,tmp);assert sha(tmp)==r['sha256'];tmp.replace(dst)
    after=hashes(Path(c.buffer_dir));assert after==allowed==json.loads((P/'coreset'/f'buffer_after_task{t}.json').read_text())['hashes']
    write(ROOT/f'buffer_after_task{t}.json',dict(hashes=after,total=len(after),seconds=time.time()-start,parent_manifest_sha256=m['parent_manifest_sha256']))
def checkpoint(a,t,next_epoch,complete=False):
    save(ROOT/'recovery/latest.pth',dict(state=capture(a),task=t,next_epoch=next_epoch,stage_complete=complete,branches=list(range(6,t+1)),contract_sha256=sha(O/'CONTRACT.json'),model_contract_sha256=sha(O/'MODEL_CONTRACT.json'),buffer_hashes=hashes(Path(a.cfg.buffer_dir))))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    verify();assert json.loads((O/'INTEGRATION.json').read_text())['status']=='PASS'
    assert not(ROOT/'TRAINING_COMPLETE.json').exists();c=cfg('train');cp=ROOT/'recovery/latest.pth'
    assert args.resume or not cp.exists();res=torch.load(cp,map_location='cpu') if args.resume else None
    dist.init_process_group('nccl');torch.cuda.set_device(0);begin=time.time()
    try:
        a,b=build(c);history=[]
        write(O/'NUMERIC_TRAIN.json',assert_parent_numeric(False))
        if res:
            assert res['contract_sha256']==sha(O/'CONTRACT.json') and res['model_contract_sha256']==sha(O/'MODEL_CONTRACT.json')
            for k in res['branches']:stage(a,k)
            a.policy.load_state_dict(res['state']['model'],strict=True);a.step=res['state']['step'];restore_rng(res['state']['rng'])
            actual=hashes(Path(c.buffer_dir));assert all(actual.get(n)==h for n,h in res['buffer_hashes'].items())
            assert actual==res['buffer_hashes'] or res['next_epoch']==100 and not res['stage_complete']
            old=json.loads((ROOT/'epochs.json').read_text());history=[r for r in old if r['task']<res['task'] or r['task']==res['task'] and r['epoch']<res['next_epoch']]
            if old!=history:write(ROOT/f'interrupted_epochs_{time.time_ns()}.json',old)
        else:
            assert not list(Path(c.buffer_dir).glob('*.pt'))
            initial=dict(rng=rng_digest(),tensors={'module.'+k:v for k,v in digest(a.policy.state_dict()).items()})
            assert initial==json.loads((P/'coreset/INITIAL_STATE.json').read_text()),'Parent initial shared tensors/RNG differ'
            write(O/'INITIAL_PUBLIC_PARITY.json',dict(passed=True,**initial))
        for t in range(6,10):
            if res and (t<res['task'] or t==res['task'] and res['stage_complete']):continue
            start=res['next_epoch'] if res and t==res['task'] else 0
            if start==100:
                store(c,t);res['stage_complete']=True;res['buffer_hashes']=hashes(Path(c.buffer_dir));save(cp,res);continue
            continuing=res and t==res['task']
            if not continuing:append(O/'BRANCH_BIRTH.jsonl',stage(a,t))
            old_digest=digest({n:v for n,v in a.policy.module.state_dict().items() if n.startswith('residual_bank.') and not n.startswith(f'residual_bank.{t}.')})
            info=build_ifa_similarity_matrix(t,c.feature_dir,c.buffer_dir);a.refs,dev=refs(c,t)
            evidence=dict(info=jsonable(info),reference_max_deviation=dev);assert evidence==json.loads((P/'coreset'/f'ifa_task{t}.json').read_text()),'IFA identity mismatch'
            write(ROOT/f'ifa_task{t}.json',evidence);write(ROOT/f'ifa_references_task{t}.json',digest({str(k):v for k,v in a.refs.items()}))
            dataset=build_dataset_for_task(c.feature_dir,t,c.buffer_dir,42,start=6)
            sampler=DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=True);loader=DataLoader(dataset,batch_size=10,num_workers=0,sampler=sampler,persistent_workers=False)
            assert len(loader)=={6:157,7:227,8:309,9:362}[t]
            # Compare logical file sequence without instantiating/iterating a second sampler.
            parent_ds=build_dataset_for_task(str(P/'features'),t,str(P/'coreset/buffer'),42,start=6) if t==6 else None
            parent_paths=sorted(str(v) for v in (P/'features'/str(t)).glob('*.pt'))
            if t>6:parent_paths=sorted(str(P/'coreset/buffer'/n) for n in json.loads((P/'coreset/storage'/f'task{t}.json').read_text())['before'])+parent_paths
            import random
            random.Random(42).shuffle(parent_paths)
            logical=[Path(v).name for v in dataset.file_paths];assert logical==[Path(v).name for v in parent_paths]
            write(ROOT/f'dataset_task{t}.json',dict(length=len(dataset),batches=len(loader),logical_order=logical,sha256=hashlib.sha256('\n'.join(logical).encode()).hexdigest(),parent_order_equal=True,sampler_seed=sampler.seed))
            a.start_task(t,loader)
            names=[n for n,p in a.policy.module.named_parameters() if p.requires_grad]
            assert {id(p) for group in a.optimizer.param_groups for p in group['params']}=={id(p) for p in a.policy.parameters() if p.requires_grad}
            write(ROOT/f'optimizer_task{t}.json',dict(names=names,kwargs=dict(c.train.optimizer.kwargs),total_steps=len(loader)*100))
            if continuing:restore(a,res['state'])
            check(a.policy.module,t,start-1)
            for epoch in range(start,100):
                sampler.set_epoch(epoch);a.policy.train();loss=0.;tic=time.time();lr=a.optimizer.param_groups[0]['lr']
                for batch in loader:loss+=a.observe(batch,t,epoch,info)
                history.append(dict(task=t,epoch=epoch,batches=len(loader),steps=a.step,loss=loss/len(loader),lr_first=lr,lr_after=a.optimizer.param_groups[0]['lr'],seconds=time.time()-tic))
                if(epoch+1)%10==0:check(a.policy.module,t,epoch,full=epoch==99)
                assert old_digest==digest({n:v for n,v in a.policy.module.state_dict().items() if n.startswith('residual_bank.') and not n.startswith(f'residual_bank.{t}.')})
                write(ROOT/'epochs.json',history)
                if epoch==99:
                    save(ROOT/'models'/f'mlrifa_{t}_model_ep99.pth',dict(state_dict=cpu(a.policy.state_dict()),cfg=c,task=t,epoch_index=99,branches=list(range(6,t+1)),modal_sha256=sha(O/'MODAL_CONTRACT.json'),model_contract_sha256=sha(O/'MODEL_CONTRACT.json'),contract_sha256=sha(O/'CONTRACT.json'),global_step=a.step))
                    append(O/'FROZEN_BRANCH_IDENTITY.jsonl',dict(task=t,old_unchanged=True,tensors=old_digest))
                checkpoint(a,t,epoch+1)
                write(O/'RUN_STATUS.json',dict(status='TRAINING',task=t,completed_epoch=epoch,optimizer_steps=a.step,updated_unix=time.time(),pid=os.getpid()))
                print(json.dumps(history[-1]),flush=True)
            store(c,t);checkpoint(a,t,100,True);res=None
        assert len(history)==400 and a.step==105500
        write(ROOT/'TRAINING_COMPLETE.json',dict(epochs=400,steps=a.step,seconds=time.time()-begin,epoch_seconds=sum(r['seconds'] for r in history),peak_allocated=torch.cuda.max_memory_allocated(),finals=hashes(ROOT/'models','*.pth')))
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
