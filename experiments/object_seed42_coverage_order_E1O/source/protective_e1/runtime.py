"""E0 update with future-index permutation and atomic epoch recovery."""
import copy
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from .bindings import bind
bind()
from .baseline_loop import E1EpochLoop
from .io import sha, write, save
from .paired_audit import buffer_pairing
from .coverage_order import CoveragePlan, CoverageOrder, OrderConfig
from libero.lifelong.algos.mlr import MLR, update_buffer_equal_quota_10, BUFFER_CAPACITY



def cpu(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy'])
    torch.set_rng_state(value['cpu']);torch.cuda.set_rng_state_all(value['cuda'])


def capture(algo):
    return dict(model=cpu(algo.policy.state_dict()),optimizer=cpu(algo.optimizer.state_dict()),
      scheduler=cpu(algo.scheduler.state_dict()),optimizer_step=algo._optimizer_step_count,
      scheduler_step=algo._scheduler_step_count,grads={n:None if p.grad is None else cpu(p.grad) for n,p in algo.policy.named_parameters()},rng=cpu(rng()),modes={n:m.training for n,m in algo.policy.named_modules()})


class OrderMLR(E1EpochLoop):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.out=Path(self.cfg.e1_root);self.out.mkdir(parents=True,exist_ok=True)
        self.recovery=self.out/'recovery/latest.pth';self.resume=None
        self.e1_start_epoch=0;self.stage=None;self.catalog={};self.verified=set()
        self.order_totals={};self.run_started=time.time();self.plan=None;self.epoch_evidence=None
        if self.recovery.exists():
            self.resume=torch.load(self.recovery,map_location='cpu')
            assert self.resume['contract_sha256']==sha(self.cfg.e1_contract)
            self.order_totals=self.resume['order_totals']
            folder=Path(self.cfg.buffer_dir).resolve()
            assert folder==self.out.resolve()/'buffer'
            expected=self.resume['buffer_hashes'];actual={p.name:p for p in folder.glob('*.pt')}
            assert set(expected)<=set(actual)
            assert all(sha(actual[k])==v for k,v in expected.items())
            extras=set(actual)-set(expected)
            if extras:
                archive=self.out/'interrupted'/('buffer_'+str(time.time_ns()));archive.mkdir(parents=True)
                for k in extras:os.replace(actual[k],archive/k)
            for name in ('lr_protocol.csv','paper_repro_train_curve.csv'):
                path=Path(self.experiment_dir)/name
                if not path.exists():continue
                with path.open() as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
                keep=[r for r in rows if int(r['stage'])<self.resume['task'] or (int(r['stage'])==self.resume['task'] and int(r['epoch'])<self.resume['next_epoch'])]
                if len(keep)!=len(rows):
                    os.replace(path,path.with_suffix('.interrupted_%d.csv'%time.time_ns()))
                    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(keep)

    def payload(self,item_id):
        row=self.catalog[item_id];path=Path(row['path'])
        if item_id not in self.verified:assert sha(path)==row['sha256'];self.verified.add(item_id)
        sample=torch.load(path,map_location='cpu')
        assert set(sample)=={'feature','actions'}
        assert all(torch.isfinite(x).all() for x in sample.values())
        return {k:v.detach().cpu() for k,v in sample.items()}

    def start_task(self,task,train_loader):
        super().start_task(task,train_loader)
        if self.resume and self.resume['task']==self.stage:
            value=self.resume['state'];self.policy.load_state_dict(value['model'],strict=True)
            self.optimizer.load_state_dict(value['optimizer']);self.scheduler.load_state_dict(value['scheduler'])
            for n,p in self.policy.named_parameters():p.grad=None if value['grads'][n] is None else value['grads'][n].to(p.device)
            self._optimizer_step_count=value['optimizer_step'];self._scheduler_step_count=value['scheduler_step']
            for n,m in self.policy.named_modules():m.training=value['modes'][n]
            self.restore_order_sidecar(self.resume['order_sidecar']);restore_rng(value['rng'])
        elif self.resume and self.resume['task']<self.stage and self.resume['stage_complete']:
            self.policy.load_state_dict(self.resume['state']['model'],strict=True);restore_rng(self.resume['state']['rng'])

    def configure_plan(self,dataset):
        paths=dataset.base.file_paths
        from libero.lifelong.algos.mlr import parse_path
        tasks=[int(parse_path(p)) for p in paths]
        ids=[('buffer/'+Path(p).name) if Path(p).name.startswith('t') else ('current/%d/%s'%(t,Path(p).name)) for p,t in zip(paths,tasks)]
        assert set(ids)==set(self.catalog)
        self.plan=CoveragePlan(tasks,ids,self.cfg.train.batch_size,self.stage)

    def ordered_batches(self,loader):
        iterator=iter(loader)  # exactly one DataLoader iterator per epoch
        while self.plan.cursor<len(self.plan.actual):
            self.order.schedule(self.plan,self.order_stage_step)
            yield next(iterator)

    def observe(self,data,current_task_id,epoch,ifa_info=None):
        assert len(data)==3
        self.plan.record_read(data[2],data[1],self.order_stage_step)
        before=self._optimizer_step_count
        loss=MLR.observe(self,[data[0],data[1]],current_task_id,epoch,ifa_info=ifa_info)
        assert self._optimizer_step_count==before+1
        self.order_stage_step+=1
        return loss

    def order_sidecar(self):
        assert not self.order.events and self.epoch_evidence is None
        return dict(stage_optimizer_step=self.order_stage_step,order=self.order.state_dict(),
            committed_epoch=self.plan.epoch if self.plan else None,
            completed_cursor=self.plan.cursor if self.plan else None)

    def restore_order_sidecar(self,state):
        self.order.load_state_dict(state['order']);self.order_stage_step=state['stage_optimizer_step']
        assert self.order.last_step is None or self.order.last_step<=self.order_stage_step

    def _buffer_hashes(self):return {p.name:sha(p) for p in sorted(Path(self.cfg.buffer_dir).glob('*.pt'))}

    def checkpoint(self,next_epoch,stage_complete=False):
        checkpoint_started=time.perf_counter()
        if self.epoch_evidence is not None:
            folder=self.out/'order_logs'/('task%d'%self.stage);folder.mkdir(parents=True,exist_ok=True)
            target=folder/('epoch%03d.json'%(next_epoch-1))
            if target.exists():os.replace(target,target.with_suffix('.interrupted_%d.json'%time.time_ns()))
            evidence=dict(self.epoch_evidence,stage=self.stage,schedules=self.order.events,
                stage_optimizer_step=self.order_stage_step)
            write(target,evidence);self.epoch_evidence=None;self.order.events=[]
        self.order_totals[str(self.stage)]=copy.deepcopy(self.order.totals)
        value=dict(state=capture(self),task=self.stage,next_epoch=next_epoch,stage_complete=stage_complete,
          order_sidecar=self.order_sidecar(),buffer_hashes=self._buffer_hashes(),order_totals=self.order_totals,
          contract_sha256=sha(self.cfg.e1_contract))
        save(self.recovery,value)
        write(self.out/'RUN_STATUS.json',dict(status='TRAINING',stage=self.stage,next_epoch=next_epoch,
          stage_optimizer_steps=self.order_stage_step,optimizer_steps=self._optimizer_step_count,order_totals=self.order_totals,
          stage_complete=stage_complete,updated_unix=time.time(),pid=os.getpid()))
        with (self.out/'checkpoint_cost.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=self.stage,next_epoch=next_epoch,stage_complete=stage_complete,seconds=time.perf_counter()-checkpoint_started,included_in_training_total=True))+'\n')

    def _append_csv(self,path,fieldnames,row):
        MLR._append_csv(path,fieldnames,row)
        if Path(path).name=='paper_repro_train_curve.csv':self.checkpoint(int(row['epoch'])+1)

    def learn_all_tasks(self,datasets,benchmark,result_summary,current_task_id):
        self.stage=current_task_id
        if self.resume and (current_task_id<self.resume['task'] or (current_task_id==self.resume['task'] and self.resume['stage_complete'])):return 0.,0.
        self.e1_start_epoch=self.resume['next_epoch'] if self.resume and self.resume['task']==current_task_id else 0
        self.catalog={};old={};new=[];self.verified=set()
        for path in sorted(Path(self.cfg.buffer_dir).glob('*.pt')):
            task=int(path.name.split('__')[0][1:]);assert task<current_task_id
            item='buffer/'+path.name;old.setdefault(task,[]).append(item)
            self.catalog[item]=dict(path=str(path),sha256=sha(path),task=task)
        for path in sorted((Path(self.cfg.feature_dir)/str(current_task_id)).glob('*.pt')):
            item='current/%d/%s'%(current_task_id,path.name);new.append(item)
            self.catalog[item]=dict(path=str(path),sha256=sha(path),task=current_task_id)
        catalog_path=self.out/('catalog_task%d.json'%current_task_id)
        if catalog_path.exists():assert json.loads(catalog_path.read_text())==self.catalog
        else:write(catalog_path,self.catalog)
        self.order = CoverageOrder(self.policy,self.cfg.seed,current_task_id,old,new,self.payload,sha(catalog_path),OrderConfig(**self.cfg.order_replay))
        self.order_stage_step=0
        if self.e1_start_epoch==self.cfg.train.n_epochs:
            self.policy.load_state_dict(self.resume['state']['model'],strict=True)
            restore_rng(self.resume['state']['rng']);self.restore_order_sidecar(self.resume['order_sidecar'])
            update_buffer_equal_quota_10(self.cfg.feature_dir,current_task_id,self.cfg.buffer_dir,
                capacity=BUFFER_CAPACITY,seed=self.cfg.seed,drop_policy='random',rate=self.cfg.past_data_rate)
            hashes=self._buffer_hashes();buffer_pairing(self.out,current_task_id,hashes)
            write(self.out/('buffer_after_task%d.json'%current_task_id),hashes)
            value=dict(self.resume,stage_complete=True,buffer_hashes=hashes);save(self.recovery,value);self.resume=value
            return 0.,0.
        result=super().learn_all_tasks(datasets,benchmark,result_summary,current_task_id)
        buffer_pairing(self.out,current_task_id,self._buffer_hashes())
        write(self.out/('buffer_after_task%d.json'%current_task_id),self._buffer_hashes())
        self.checkpoint(self.cfg.train.n_epochs,True)
        self.resume=None
        return result
