"""Original MLR+IFA observe and buffer functions; E1 wrapper and epoch recovery."""
import copy
import csv
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate

from .bindings import bind
bind()
from .baseline_loop import E1EpochLoop
from .control import Controller, Settings, rng_state, restore_rng
from .io import write, save, sha
from libero.lifelong.algos.mlr import MLR, FileTrajectoryDataset, update_buffer_equal_quota_10, BUFFER_CAPACITY


def cpu(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k:cpu(v) for k,v in x.items()}
    if isinstance(x, list): return [cpu(v) for v in x]
    if isinstance(x, tuple): return tuple(cpu(v) for v in x)
    return copy.deepcopy(x)


def capture(algo):
    return dict(model=cpu(algo.policy.state_dict()), optimizer=cpu(algo.optimizer.state_dict()),
                scheduler=copy.deepcopy(algo.scheduler.state_dict()) if algo.scheduler else None,
                optimizer_step=algo._optimizer_step_count, scheduler_step=algo._scheduler_step_count,
                rng=cpu(rng_state()), modes={n:m.training for n,m in algo.policy.named_modules()})


def restore(algo, state):
    algo.policy.load_state_dict(state['model'], strict=True)
    algo.optimizer.load_state_dict(state['optimizer'])
    if algo.scheduler: algo.scheduler.load_state_dict(state['scheduler'])
    algo._optimizer_step_count=state['optimizer_step']; algo._scheduler_step_count=state['scheduler_step']
    for n,m in algo.policy.named_modules(): m.training=state['modes'][n]
    restore_rng(state['rng'])


class E1MLR(E1EpochLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resume = None
        self.e1_stage = None
        self.e1_start_epoch = 0
        self.report = Path(self.cfg.e1_report)
        self.recovery = Path(self.cfg.e1_recovery)
        self.controller = None
        self.e1_totals = dict(checks=0, triggered=0, accepted=0, capped=0, overhead_seconds=0.)
        if self.recovery.exists():
            self.resume = torch.load(self.recovery, map_location='cpu')
            assert self.resume['protocol_sha256'] == sha(self.cfg.e1_protocol), 'E1_PROTOCOL_MISMATCH'
            self.e1_totals = self.resume['totals']
            # A stage-end buffer transaction may have been interrupted after epoch99.
            # Restore membership from the authoritative checkpoint, only inside E1's own buffer.
            directory=Path(self.cfg.buffer_dir).resolve()
            assert directory == Path(self.cfg.e1_root).resolve()/'buffer'
            expected=set(self.resume['buffer_files'])
            actual={p.name for p in directory.glob('*.pt')}
            assert expected <= actual, 'E1_RECOVERY_BUFFER_MISSING'
            for name in actual-expected: (directory/name).unlink()
            # CSV writes precede atomic checkpoint replacement: retain committed epochs.
            for name in ('lr_protocol.csv','paper_repro_train_curve.csv'):
                p=Path(self.experiment_dir)/name
                if p.exists():
                    with p.open() as f:
                        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
                    keep=[r for r in rows if int(r['stage'])<self.resume['task'] or
                          (int(r['stage'])==self.resume['task'] and int(r['epoch'])<self.resume['next_epoch'])]
                    if len(keep)!=len(rows):
                        n=1
                        while p.with_suffix('.interrupted%d.csv'%n).exists():n+=1
                        p.rename(p.with_suffix('.interrupted%d.csv'%n))
                        with p.open('w',newline='') as f:
                            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(keep)

    def start_task(self, task, train_loader):
        super().start_task(task, train_loader)
        model=getattr(self.policy, 'module', self.policy)
        self.controller=Controller(model,self.cfg.seed,Settings())
        if self.resume:
            if self.resume['task']==self.e1_stage:
                restore(self,self.resume['state'])
                self.controller.sampling_rng.setstate(self.resume['check_rng'])
            elif self.resume['task']<self.e1_stage and self.resume['stage_complete']:
                self.policy.load_state_dict(self.resume['state']['model'],strict=True)
                restore_rng(self.resume['state']['rng'])
                # New task starts its own seeded check stream, as on uninterrupted runs.
            else: raise RuntimeError('Unexpected E1 restore stage')

    def observe(self, data, current_task_id, epoch, ifa_info=None):
        def sample(rng, count):
            if not self.old_files: return None, []
            indices=rng.sample(range(len(self.old_files)),min(count,len(self.old_files)))
            ds=FileTrajectoryDataset([self.old_files[i] for i in indices])
            batch=default_collate([ds[i] for i in range(len(ds))])
            assert all(int(t)<current_task_id for t in batch[1])
            return self.map_tensor_to_device(batch[0]), [Path(self.old_files[i]).name for i in indices]
        native=lambda: MLR.observe(self,data,current_task_id,epoch,ifa_info=ifa_info)
        loss,row=self.controller.update(native,lambda:self._optimizer_step_count,sample,
                    dict(seed=self.cfg.seed,task=current_task_id,epoch=epoch))
        if not math.isfinite(float(loss)): raise FloatingPointError('Nonfinite base training loss')
        if row:
            self.e1_totals['checks']+=1
            for k in ['triggered','accepted','capped']: self.e1_totals[k]+=int(row[k])
            self.e1_totals['overhead_seconds']+=row['overhead_seconds']
            # Per-epoch event stream is truncated to the checkpoint boundary on resume.
            path=self.report/'events'/('task%d_epoch%03d.jsonl'%(current_task_id,epoch))
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        return loss

    def checkpoint(self, next_epoch, stage_complete=False):
        value=dict(state=capture(self),task=self.e1_stage,next_epoch=next_epoch,
                   stage_complete=stage_complete,check_rng=self.controller.sampling_rng.getstate(),
                   totals=self.e1_totals.copy(),buffer_files=sorted(p.name for p in Path(self.cfg.buffer_dir).glob('*.pt')),
                   protocol_sha256=sha(self.cfg.e1_protocol))
        save(self.recovery,value)
        write(self.report/'STATUS.json',dict(status='TRAINING',seed=self.cfg.seed,task=self.e1_stage,
              next_epoch=next_epoch,stage_complete=stage_complete,optimizer_step=self._optimizer_step_count,
              scheduler_step=self._scheduler_step_count,totals=self.e1_totals,
              recovery=str(self.recovery),e1_training_active=True,baseline_training_rerun=False))

    def _append_csv(self, path, fieldnames, row):
        MLR._append_csv(path,fieldnames,row)
        if Path(path).name=='paper_repro_train_curve.csv': self.checkpoint(int(row['epoch'])+1)

    def learn_all_tasks(self,datasets,benchmark,result_summary,current_task_id):
        self.e1_stage=current_task_id
        if self.resume and (current_task_id<self.resume['task'] or
                (current_task_id==self.resume['task'] and self.resume['stage_complete'])):
            return 0.,0.
        self.e1_start_epoch=self.resume['next_epoch'] if self.resume and self.resume['task']==current_task_id else 0
        self.old_files=sorted(str(p) for p in Path(self.cfg.buffer_dir).glob('*.pt'))
        assert all(int(Path(p).name.split('__')[0][1:])<current_task_id for p in self.old_files)
        # Preserve evidence from interrupted partial epochs, then redo from last full epoch.
        events=self.report/'events'
        if events.exists():
            for p in events.glob('task%d_epoch*.jsonl'%current_task_id):
                if '.attempt' in p.name: continue
                if int(p.stem.split('epoch')[1])>=self.e1_start_epoch:
                    n=1
                    while p.with_suffix('.attempt%d.jsonl'%n).exists(): n+=1
                    os.replace(p,p.with_suffix('.attempt%d.jsonl'%n))
        write(self.report/('BUFFER_BEFORE_TASK%d.json'%current_task_id),dict(task=current_task_id,files=self.old_files))
        if self.e1_start_epoch==self.cfg.train.n_epochs:
            # Epoch99 was saved but end-task buffer transaction was not committed.
            self.policy.load_state_dict(self.resume['state']['model'],strict=True)
            restore_rng(self.resume['state']['rng'])
            self.controller=Controller(getattr(self.policy,'module',self.policy),self.cfg.seed)
            self.controller.sampling_rng.setstate(self.resume['check_rng'])
            update_buffer_equal_quota_10(self.cfg.feature_dir,current_task_id,self.cfg.buffer_dir,
                capacity=BUFFER_CAPACITY,seed=self.cfg.seed,drop_policy='random',rate=self.cfg.past_data_rate)
            value=dict(self.resume,stage_complete=True,buffer_files=sorted(p.name for p in Path(self.cfg.buffer_dir).glob('*.pt')))
            save(self.recovery,value);self.resume=value
            return 0.,0.
        result=super().learn_all_tasks(datasets,benchmark,result_summary,current_task_id)
        self.checkpoint(self.cfg.train.n_epochs,True)
        self.resume=None
        return result
