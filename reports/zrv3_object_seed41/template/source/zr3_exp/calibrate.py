import argparse,shutil
from .diagnostics import *
from torch.utils.data import DataLoader,DistributedSampler
from libero.lifelong.algos.mlr import build_dataset_for_task

def store(t):
    src=P/PARM/'storage'/f'task{t}.json';m=js(src);target=O/'storage'/f'task{t}.json'
    if target.exists():assert js(target)['parent_manifest_sha256']==sha(src)
    else:write(target,dict(parent_manifest_sha256=sha(src),selected=m['selected'],task=t))
    directory=O/'buffer';directory.mkdir(exist_ok=True)
    for r in m['selected']:
        s=Path(r['source']);dst=directory/r['destination'];assert sha(s)==r['sha256']
        if dst.exists():assert sha(dst)==r['sha256']
        else:
            tmp=dst.with_suffix('.copying');shutil.copy2(s,tmp);assert sha(tmp)==r['sha256'];tmp.replace(dst)
    actual={p.name:sha(p) for p in sorted(directory.glob('*.pt'))}
    expected=js(P/PARM/f'buffer_after_task{t}.json')['hashes'];assert actual==expected
    write(O/f'buffer_after_task{t}.json',dict(hashes=actual,parent_manifest_sha256=sha(src),count=len(actual)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    verify();assert js(O/'INTEGRATION.json')['status']=='PASS';assert not (O/'CALIBRATED.json').exists()
    cp=O/'recovery/latest.pth';assert args.resume or not cp.exists(),'Explicit resume required'
    setup();p,t0=new_public();t0_hash=tensor_hash(t0);assert t0_hash==js(O/'INTEGRATION.json')['T0_hash']
    (O/'buffer').mkdir(exist_ok=True);res=torch.load(cp,map_location='cpu') if args.resume else None
    if res:assert res['contract_sha256']==sha(O/'CONTRACT.json')
    else:assert not list((O/'buffer').glob('*.pt'))
    previous=None;epochs=[];globalstep=0;exposures=0
    if res:
        previous,_=make_bank(res['task']);previous.load_state_dict(res['bank'],strict=True)
        actual={f.name:sha(f) for f in (O/'buffer').glob('*.pt')}
        assert all(actual.get(n)==h for n,h in res['buffer_hashes'].items())
        if actual!=res['buffer_hashes']:assert res['next_epoch']==10,'Uncommitted buffer outside stage boundary'
        epochs=res['epochs'];globalstep=res['globalstep'];exposures=res['exposures']
        # Committed epoch checkpoint is authority; all raw batch attempts remain append-only.
        if (O/'epochs.json').exists() and js(O/'epochs.json')!=epochs:write(O/f'interrupted_epochs_{time.time_ns()}.json',js(O/'epochs.json'))
    for t in range(6,10):
        if res and t<res['task']:continue
        if res and t==res['task'] and res['next_epoch']==10:
            store(t);res=None;continue
        parent=load_stage(p,t);ph=tensor_hash(p)
        bank,geometry=make_bank(t,previous)
        dataset=build_dataset_for_task(str(P/'features'),t,str(O/'buffer'),SEED,start=6)
        assert len(dataset)==LENS[t-6]
        sampler=DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=True,seed=0)
        loader=DataLoader(dataset,batch_size=10,sampler=sampler,num_workers=0,drop_last=False)
        assert len(loader)==BATCHES[t-6]
        order=[Path(f).name for f in dataset.file_paths];write(O/f'dataset_stage{t}.json',dict(order=order,sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest()))
        write(O/f'gates_stage{t}.json',dict(geometry=geometry,canonical_prefix=list(range(t+1)),FP64=True))
        opt,sch=optimizer(bank,10*len(loader));start_epoch=0
        if res:
            bank.load_state_dict(res['bank'],strict=True);opt.load_state_dict(res['optimizer']);sch.load_state_dict(res['scheduler'])
            for n,v in bank.named_parameters():v.grad=None if res['grads'][n] is None else res['grads'][n].cuda()
            for n,m in bank.named_modules():m.training=res['modes'][n]
            assert ph==res['public_hash'] and t0_hash==res['T0_hash'];restore_rng(res['rng']);start_epoch=res['next_epoch'];res=None
        check(p,t0,bank,dataset,'stage_start',ph,t0_hash)
        for epoch in range(start_epoch,10):
            sampler.set_epoch(epoch);tic=time.time();losses=[];firstlr=opt.param_groups[0]['lr'];sample_order=list(iter(sampler));offset=0
            for bi,(samples,tasks) in enumerate(loader):
                ids=torch.tensor([int(x) for x in tasks],device='cuda');h=samples['feature'].squeeze(1).cuda();a=samples['actions'].squeeze(1).cuda()
                row=update(p,t0,bank,opt,sch,h,a,ids);globalstep+=1;exposures+=len(h);losses.append(row['loss'])
                indices=sample_order[offset:offset+len(h)];offset+=len(h)
                append(O/'calibration_batches.jsonl',dict(stage=t,epoch=epoch,batch=bi,logical_step=globalstep,indices=indices,samples=[order[i] for i in indices],**row))
            assert offset==len(dataset)
            torch.cuda.synchronize();seconds=time.time()-tic
            branch_steps={n:int(opt.state[b.U].get('step',0)) for n,b in bank.branches.items()}
            row=dict(task=t,epoch=epoch,batches=len(loader),samples=len(dataset),optimizer_steps=globalstep,exposures=exposures,loss=float(np.mean(losses)),lr_first=firstlr,lr_after=opt.param_groups[0]['lr'],branch_steps=branch_steps,seconds=seconds)
            if epoch in (0,9):check(p,t0,bank,dataset,f'epoch{epoch}',ph,t0_hash)
            epochs.append(row);write(O/'epochs.json',epochs)
            snapshot=dict(task=t,next_epoch=epoch+1,bank=cpu(bank.state_dict()),optimizer=cpu(opt.state_dict()),scheduler=cpu(sch.state_dict()),grads={n:cpu(v.grad) for n,v in bank.named_parameters()},modes={n:m.training for n,m in bank.named_modules()},rng=cpu(rng()),globalstep=globalstep,exposures=exposures,epochs=epochs,parent=parent,public_hash=ph,T0_hash=t0_hash,contract_sha256=sha(O/'CONTRACT.json'),dataset_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),buffer_hashes={f.name:sha(f) for f in (O/'buffer').glob('*.pt')})
            if epoch==9:save(O/'models'/f'zrv3_stage{t}_calibration_ep9.pth',snapshot)
            save(cp,snapshot);status('CALIBRATING',task=t,epoch=epoch,optimizer_steps=globalstep,exposures=exposures);print(json.dumps(row),flush=True)
        check(p,t0,bank,dataset,'stage_end',ph,t0_hash);store(t);previous=bank
    assert len(epochs)==40 and globalstep==STEPS and exposures==EXPOSURES
    write(O/'CALIBRATED.json',dict(status='CALIBRATED',epochs=40,optimizer_steps=globalstep,exposures=exposures,checkpoints={f.name:sha(f) for f in (O/'models').glob('*.pth')},summed_epoch_seconds=sum(r['seconds'] for r in epochs),T0_hash=t0_hash,gpu_peak_allocated=torch.cuda.max_memory_allocated()))
    status('CALIBRATED')
if __name__=='__main__':main()
