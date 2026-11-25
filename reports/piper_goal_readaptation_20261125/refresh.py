"""Readapt the shared GOAL base to the archived PiPER recapture."""
import hashlib
import json
import os
import random
import subprocess
import sys
from decimal import Decimal
from pathlib import Path


def inspect(work, out):
    for parent in (work.parent.parent/'datasets',work.parent/'datasets',work/'batches',work.parent/'diagnostics'):
        if parent.is_dir():
            print(str(parent), sorted(p.name for p in parent.iterdir())[-65:])
            if parent.name=='datasets':
                for directory in sorted(p for p in parent.iterdir() if p.is_dir()):
                    print(directory.name,sorted(x.name for x in directory.iterdir())[-32:])
    batch=work/'batches/B1_20261123'
    ranking=json.loads((work/'offline_diagnostics/b1_ode_lerobot_20261123/selection.json').read_bytes())
    e=ranking['ranking'][0]
    r=(batch/e['source'])
    print('example',e['frames'],e['first'],r.resolve(),(work/'recordings'/e['id']).resolve())
    print('row',json.loads((r/'alignment.jsonl').open().readline()).keys())
    print('manifest',list(json.loads((batch/'manifest.json').read_bytes())))
    print('base batch record',str(json.loads((batch/'manifest.json').read_bytes())['episodes'][0])[:700])
    rr=[json.loads(s) for s in (r/'alignment.jsonl').read_bytes().splitlines()]
    print('raw count',len(rr),'row example',str(rr[0])[:900],'last target',str(rr[-1]['target'])[:140])
    print('all files',sorted(x.name for x in r.iterdir()))
    root=work.parent.parent/'datasets/piper_real'
    for d in sorted(p for p in root.glob('*') if p.is_dir()):
        print('station repository',d.name,sorted(x.name for x in d.iterdir())[-12:])
    for role in ('overhead','wrist'):
        p=r/'episode0/camera/color'/role
        print(role,len(list(p.iterdir())),min(q.name for q in p.iterdir()))


SELECTION='130b51e723fe9839ed6986f0d9bcc7b8035e49ffe103cd8847b837607e1227b9'
OLD_BEST='cadfb14f1ee513f42ad7b68169f1a12f23dfcd675273a15387c572eb3b840d22'
EPOCHS=12


def read(path):
    return json.loads(path.read_bytes())


def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=1)+'\n')
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def episode_data(work,row):
    import numpy as np
    path=work/row['source']
    raw=(path/'alignment.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==row['alignment_sha256']
    lines=[json.loads(line) for line in raw.splitlines()]
    lo,n=row['first'],row['frames']
    assert len(lines)==lo+n and all(r['target'] is None for r in lines[:lo])
    stamps=np.array([r['sample_monotonic_ns'] for r in lines],dtype=np.int64)
    assert np.all(np.diff(stamps)>0)
    cameras=[]
    for name in ('overhead','wrist'):
        files=sorted((path/'episode0/camera/color'/name).glob('*.jpg'),key=lambda p:Decimal(p.stem))
        assert len(files)==len(lines) and all(abs(int(Decimal(p.stem)*10**9)-t)<=500 for p,t in zip(files,stamps)),row['id']
        cameras.append(files[lo:])
    assert [p.stem for p in cameras[0]]==[p.stem for p in cameras[1]]
    q,a=[np.asarray([r[k]['joints_rad']+[r[k]['gripper_mm']] for r in lines[lo:]],dtype=np.float32) for k in ('feedback','target')]
    assert q.shape==a.shape==(n,7) and np.isfinite(q).all() and np.isfinite(a).all()
    return dict(id=row['id'],task=row['task'],split=row['split'],q=q,a=a,cameras=cameras)


def prepare(work,out):
    import piper_base as base
    old=work/'offline_training/b123_goal_v1'
    ranking=work/'offline_diagnostics/b1_ode_lerobot_20261123/selection.json'
    assert sha(ranking)==SELECTION and sha(old/'best.pt')==OLD_BEST
    rank=read(ranking); selected=set(rank['selected_ids'])
    batch=work/'batches/B1_20261123'
    meta=read(batch/'manifest.json')
    successes={r['episode']:r for r in meta['episodes'] if r['outcome']=='success'}
    failures={r['episode'] for r in meta['episodes'] if r['outcome']!='success'}
    assert len(selected)==50 and len(successes)==84 and len(failures)==7
    old_split=read(old/'split.json')
    available={r['id']:r for r in rank['ranking']}
    assert set(available)==set(successes) and selected<=set(available)
    rows=[]
    for name in sorted(selected):
        r=available[name]
        assert r['source']=='success/'+name and successes[name]['sha256']==r['source_sha256']
        assert name not in old_split
        rows.append(dict(id=name,task=0,split='train',
             source=str((batch/r['source']).relative_to(work)),first=r['first'],frames=r['frames'],
             alignment_sha256=r['alignment_sha256']))
    old_b1=read(work/'expert_sets'/base.SETS[0][0]/'manifest.json')
    assert not ({r['alignment_sha256'] for r in rows}&{e['alignment_sha256'] for e in old_b1['episodes']})
    for task in (1,2):
        set_name,expected,_=base.SETS[task]
        path=work/'expert_sets'/set_name/'manifest.json'
        assert sha(path)==expected
        for e in sorted(read(path)['episodes'],key=lambda e:e['episode_id']):
            name=e['episode_id']
            assert (path.parent/e['episode_dir']).resolve()==(work/'recordings'/name).resolve()
            rows.append(dict(id=name,task=task,split='train',source='recordings/'+name,
                first=e['first_action_row'],frames=e['action_count'],alignment_sha256=e['alignment_sha256']))
    assert len({r['id'] for r in rows})==len({r['alignment_sha256'] for r in rows})==150
    for task in range(3):
        assert sum(r['task']==task and r['split']=='train' for r in rows)==50
    for asset,expected in [('goal_multitask_model_ep96.pth',base.GOAL_SHA),('clip_tokenizer_goal.json',base.TOKEN_SHA)]:
        p=out/asset
        if not p.exists():p.symlink_to(old/asset)
        assert sha(p)==expected
    manifest=dict(recapture='B1_20261123',selection_sha256=SELECTION,batch_manifest_sha256=sha(batch/'manifest.json'),
        prior_best_sha256=OLD_BEST,excluded_failures=sorted(failures),remaining_unselected_success_count=len(successes)-len(selected),episodes=rows)
    if (out/'manifest.json').exists():assert manifest==read(out/'manifest.json')
    else:write(out/'manifest.json',manifest)
    print('manifest frozen',len(rows),'episodes',sha(out/'manifest.json'),flush=True)


def initialize(out):
    import cv2
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    import piper_base as base
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(base.SEED);np.random.seed(base.SEED);random.seed(base.SEED)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    source=out/'goal_multitask_model_ep96.pth'
    tokens=out/'clip_tokenizer_goal.json'
    assert sha(source)==base.GOAL_SHA and sha(tokens)==base.TOKEN_SHA
    with torch.serialization.safe_globals([(base.EasyDict,'easydict.EasyDict')]):
        loaded=torch.load(source,map_location='cpu',weights_only=True)
    state={k.removeprefix('module.'):v for k,v in loaded['state_dict'].items()}
    cfg=loaded['cfg']['policy']
    assert cfg['rank']==8
    scale=cfg['alpha']/cfg['rank']
    return state,base.Frontend(state,scale,Tokenizer.from_file(str(tokens))).cuda(),scale


def cache_id(base,out):
    return sha(Path(base.__file__))+base.GOAL_SHA+base.TOKEN_SHA+sha(out/'manifest.json')


def fixed_epochs(base,model,frames,out,config):
    import torch
    from torch import nn
    n=len(frames.a)
    weights=n/(3*torch.bincount(frames.tag,minlength=3).float())
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=config['learning_rate'],weight_decay=.01)
    history=[]
    latest=out/'resume.pt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        assert saved['config']==config
        model.load_state_dict(saved['policy'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        history=saved['history']
        del saved
    for epoch in range(len(history)+1,config['epochs']+1):
        model.train()
        torch.manual_seed(base.SEED+epoch)
        loss_sum=0.
        updates=0
        for ids in torch.randperm(n,device='cuda').split(config['batch_size']):
            batch=frames.batch(ids)
            optimizer.zero_grad(set_to_none=True)
            nll,_=base.compute(model,batch)
            loss=(nll*weights[batch[2]]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            loss_sum+=loss.item()*len(ids)
            updates+=1
        history.append(dict(epoch=epoch,updates=updates,rows=n,train_nll=loss_sum/n))
        base.save_pt(latest,dict(config=config,history=history,policy=model.state_dict(),optimizer=optimizer.state_dict()))
        write(out/'history.json',history)
        print('epoch',epoch,'of',EPOCHS,'updates',updates,'train loss',loss_sum/n,flush=True)
    base.save_pt(out/'final.pt',dict(config=config,epoch=EPOCHS,policy=model.state_dict()))
    write(out/'finished.json',dict(epochs=EPOCHS,rows_per_epoch=n,updates=sum(e['updates'] for e in history),
          checkpoint_sha256=sha(out/'final.pt'),selection='fixed final epoch',validation_episodes=0))


def train(work,out):
    import fcntl
    import torch
    import piper_base as base
    lock=(out/'training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_NB|fcntl.LOCK_EX)
    manifest=read(out/'manifest.json')
    rows=manifest['episodes']
    assert len(rows)==150 and all(r['split']=='train' for r in rows)
    jobs=[episode_data(work,r) for r in rows]
    print('verified archived observations and targets',len(jobs),flush=True)
    state,front,scale=initialize(out)
    stats=base.fit_stats(jobs)
    model=base.Policy(state,front.language.detach().cpu(),stats).cuda()
    del state
    base.save_json(out/'split.json',{r['id']:r['split'] for r in rows})
    base.save_json(out/'normalization.json',stats)
    config=dict(seed=base.SEED,history=base.HISTORY,epochs=EPOCHS,batch_size=128,learning_rate=1e-4,
         checkpoint_sha256=base.GOAL_SHA,tokenizer_sha256=base.TOKEN_SHA,script_sha256=sha(Path(base.__file__)),
         adapter_sha256=sha(Path(__file__)),manifest_sha256=sha(out/'manifest.json'),normalization=stats,
         checkpoint_rule='fixed final epoch, all selected examples train',validation_episodes=0,
         split_sha256=sha(out/'split.json'),language=[r[2] for r in base.SETS],lora_scale=scale,
         observations='overhead,wrist,feedback joints (radians),gripper (mm)',
         action='absolute recorded joint and gripper target via standardized target minus feedback',
         readout='last state token after both current camera tokens',device=torch.cuda.get_device_name())
    write(out/'config.json',config)
    write(out/'smoke.json',base.verify_tasks(front,model,jobs))
    features=base.get_features(front,jobs,out/'visual_cache',cache_id(base,out))
    del front
    frames=base.Frames(jobs,features)
    del features
    print('formal start',len(frames.a),flush=True)
    fixed_epochs(base,model,frames,out,config)


def launch(work,out):
    import fcntl
    with (out/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'finished.json').exists():
            print('training already finished');return
        if (out/'pid').exists():
            proc=Path('/proc')/(out/'pid').read_text().strip()/'cmdline'
            if proc.exists() and str(out/'refresh.py').encode() in proc.read_bytes():
                print('training active');return
        with (out/'training.log').open('ab') as log:
            p=subprocess.Popen([sys.executable,'-Bu',str(out/'refresh.py'),str(work),str(out),'train'],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=out)
        (out/'pid').write_text(str(p.pid))
        print('offline process',p.pid,flush=True)


def status(out):
    p=out/'training.log'
    if p.exists():print(p.read_text(errors='replace')[-1700:])
    print('finished',(out/'finished.json').exists(),'verified',(out/'verification.json').exists())


def verify(work,out):
    import numpy as np
    import torch
    import piper_base as base
    meta=read(out/'finished.json')
    ckpt=out/'final.pt'
    assert sha(ckpt)==meta['checkpoint_sha256'] and meta['epochs']==EPOCHS
    saved=torch.load(ckpt,map_location='cpu',weights_only=True)
    cfg=saved['config']
    assert cfg==read(out/'config.json') and saved['epoch']==EPOCHS and cfg['validation_episodes']==0
    assert cfg['adapter_sha256']==sha(Path(__file__)) and cfg['script_sha256']==sha(Path(base.__file__))
    assert cfg['manifest_sha256']==sha(out/'manifest.json') and cfg['split_sha256']==sha(out/'split.json')
    rows=read(out/'manifest.json')['episodes']
    assert {r['id']:'train' for r in rows}==read(out/'split.json')
    jobs=[episode_data(work,r) for r in rows]
    assert base.fit_stats(jobs)==saved['config']['normalization']
    n=sum(len(e['q']) for e in jobs)
    history=read(out/'history.json')
    updates=(n+cfg['batch_size']-1)//cfg['batch_size']
    assert len(history)==EPOCHS and meta['updates']==updates*EPOCHS
    assert all(r['epoch']==i+1 and r['rows']==n and r['updates']==updates and np.isfinite(r['train_nll']) for i,r in enumerate(history))
    state,front,_=initialize(out)
    model=base.Policy(state,front.language.detach().cpu(),cfg['normalization'])
    model.load_state_dict(saved['policy'],strict=True)
    model.cuda().eval()
    del state,saved
    samples=[]
    with torch.inference_mode():
        for task in range(3):
            ep=next(e for e in jobs if e['task']==task)
            end=len(ep['q'])//2
            ix=list(range(end-base.HISTORY+1,end+1))
            v=front.encode(ep['cameras'],ix,task)[None].cuda()
            q=torch.from_numpy(ep['q'][ix])[None].cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                _,pred=model(v,q,torch.tensor([task],device='cuda'))
            assert pred.shape==(1,7) and torch.isfinite(pred).all()
            err=pred[0].cpu().numpy()-ep['a'][end]
            samples.append(dict(task='B'+str(task+1),episode=ep['id'],row=end,joint_mae_deg=float(np.abs(np.rad2deg(err[:6])).mean()),
                                gripper_mae_mm=float(abs(err[-1]))))
    del front
    vs={}
    for ep in jobs:
        record=torch.load(out/'visual_cache'/(ep['id']+'.pt'),weights_only=True,map_location='cpu')
        assert record['identity']==cache_id(base,out)
        vs[ep['id']]=record['vision']
    frames=base.Frames(jobs,vs)
    metrics,score=base.evaluate(model,frames,cfg['batch_size'])
    assert all(m['rows']==sum(len(e['q']) for e in jobs if 'B'+str(e['task']+1)==k) for k,m in metrics.items())
    assert all(np.isfinite(float(v)) for m in metrics.values() for v in m.values())
    assert sha(work/'offline_training/b123_goal_v1/best.pt')==OLD_BEST
    report=dict(final_sha256=meta['checkpoint_sha256'],raw_training_windows=samples,
            training_metrics=metrics,training_macro_normalized_mae=score,scope='training-set fit and checkpoint reload')
    write(out/'verification.json',report)
    print('raw RGB checkpoint verification',samples,'training metrics',metrics,flush=True)


def evidence(out):
    manifest=read(out/'manifest.json')
    rows=manifest['episodes']
    report=dict(training={f'B{i+1}':dict(episodes=sum(r['task']==i for r in rows),
               action_rows=sum(r['frames'] for r in rows if r['task']==i)) for i in range(3)},
               training_episodes=len(rows),validation_episodes=0,
               recapture_success_count=84,recapture_selected_count=50,recapture_failures=7,
               old_base_sha256=OLD_BEST,manifest_sha256=sha(out/'manifest.json'),
               finished=read(out/'finished.json'),verification=read(out/'verification.json'),
               smoke=read(out/'smoke.json'))
    write(out/'evidence.json',report)
    print('evidence',report['training'],report['finished'],flush=True)


if __name__=='__main__':
    work,out=map(Path,sys.argv[1:3])
    action=sys.argv[3]
    if action=='inspect':inspect(work,out)
    elif action=='prepare':prepare(work,out)
    elif action=='launch':launch(work,out)
    elif action=='train':train(work,out)
    elif action=='status':status(out)
    elif action=='verify':verify(work,out)
    elif action=='evidence':evidence(out)
