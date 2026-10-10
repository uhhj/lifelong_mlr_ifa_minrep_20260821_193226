"""E1-O: mutable future indices, complete coverage, no replacement hook.

This module's plan/permutation machinery needs only the standard library.
Torch and the inherited scorer are imported only when real scoring is invoked.
"""
import copy
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class OrderConfig:
    enabled: bool = True
    refresh_interval_steps: int = 500
    refresh_at_stage_start: bool = True
    max_interval_steps: int = 500
    cross_epoch: bool = False
    max_candidates_per_old_task: int = 64
    reference_per_old_task: int = 16
    current_reference_size: int = 16
    gradient_scope: str = 'policy_head'
    conflict_coefficient: float = 1.0
    softmax_temperature: float = 1.0
    numerical_epsilon: float = 1e-8
    permutation: str = 'weighted_without_replacement'
    preserve_task_slots: bool = True
    pin_first_slot: bool = True
    pin_incomplete_last_batch: bool = True
    fallback: str = 'original_order'

    def __post_init__(self):
        expected = dict(refresh_interval_steps=500,refresh_at_stage_start=True,
            max_interval_steps=500,cross_epoch=False,max_candidates_per_old_task=64,
            reference_per_old_task=16,current_reference_size=16,gradient_scope='policy_head',
            conflict_coefficient=1.,softmax_temperature=1.,numerical_epsilon=1e-8,
            permutation='weighted_without_replacement',preserve_task_slots=True,
            pin_first_slot=True,pin_incomplete_last_batch=True,fallback='original_order')
        assert all(getattr(self,k)==v for k,v in expected.items())


def weighted_permutation(items, scores, rng, tau=1.):
    """Sequential Plackett-Luce draws, removing each selected candidate."""
    assert len(items)==len(scores) and len(set(items))==len(items)
    assert all(math.isfinite(s) for s in scores) and tau>0
    remaining=list(range(len(items)));chosen=[]
    while remaining:
        peak=max(scores[i]/tau for i in remaining)
        weights=[math.exp(scores[i]/tau-peak) for i in remaining]
        i=rng.choices(remaining,weights=weights,k=1)[0]
        chosen.append(i);remaining.remove(i)
    return [items[i] for i in chosen],chosen


class CoveragePlan:
    def __init__(self, tasks, ids, batch_size, current_task):
        self.tasks=list(map(int,tasks));self.ids=list(ids);self.batch_size=batch_size
        self.current_task=current_task;self.base=[];self.actual=[];self.cursor=0
        self.epoch=None;self.claimed=set();self.read_rows=[];self.last_batch=None
        assert len(set(self.ids))==len(self.ids)==len(self.tasks)

    def begin_epoch(self, indices, epoch):
        assert self.cursor==len(self.actual),'Previous epoch not fully consumed'
        self.base=list(indices);self.actual=list(indices);self.cursor=0;self.epoch=epoch
        self.claimed=set();self.read_rows=[];self.last_batch=None
        assert Counter(self.base)==Counter(range(len(self.ids))), 'Single rank full traversal required'

    def __len__(self):return (len(self.ids)+self.batch_size-1)//self.batch_size

    def pinned(self,p):
        n=len(self.actual);b=self.batch_size
        return p%b==0 or (n%b!=0 and p>=n-n%b)

    def verify(self, candidate=None, prefix=None):
        a=self.actual if candidate is None else candidate
        assert Counter(a)==Counter(self.base),'ID multiplicity changed'
        assert a[:self.cursor]==(self.actual if prefix is None else prefix)[:self.cursor], 'Used prefix changed'
        for p,(original,actual) in enumerate(zip(self.base,a)):
            assert self.tasks[original]==self.tasks[actual], 'Task slot changed'
            if self.tasks[original]==self.current_task or self.pinned(p):
                assert actual==original,'Protected identity changed'

    def candidates(self, stage_step, rng, limit=64):
        b=self.batch_size;assert self.cursor%b==0
        end=min(len(self.actual),self.cursor+(500-stage_step%500)*b)
        selected={}
        for task in sorted(set(self.tasks)-{self.current_task}):
            available=[p for p in range(self.cursor,end) if not self.pinned(p)
                       and self.tasks[self.actual[p]]==task and p not in self.claimed]
            positions=sorted(rng.sample(available,min(limit,len(available))))
            selected[task]=positions
        return selected,end

    def commit(self,positions,ordered,end):
        before=list(self.actual)
        try:
            assert len(set(positions))==len(positions)==len(ordered)
            assert all(self.cursor<=p<end and p not in self.claimed and not self.pinned(p) for p in positions)
            assert Counter(before[p] for p in positions)==Counter(ordered)
            trial=list(before)
            for p,i in zip(positions,ordered):trial[p]=i
            self.verify(trial,before)
        except AssertionError as error:
            assert self.actual==before
            return False,str(error)
        self.actual=trial;self.claimed.update(positions)
        return True,None

    def record_read(self, identities, tasks, stage_step):
        start,end=self.last_batch
        indices=self.actual[start:end];expected=[self.ids[i] for i in indices]
        assert list(identities)==expected,'DataLoader delivered IDs different from plan'
        assert list(map(int,tasks))==[self.tasks[i] for i in indices]
        self.read_rows.append(dict(step=stage_step,start=start,end=end,
            original_ids=[self.ids[i] for i in self.base[start:end]],actual_ids=list(identities),
            task_ids=list(map(int,tasks))))

    def finish(self):
        assert self.cursor==len(self.actual) and len(self.read_rows)==len(self)
        self.verify()
        actual=[x for row in self.read_rows for x in row['actual_ids']]
        original=[self.ids[i] for i in self.base]
        assert Counter(actual)==Counter(original)==Counter(self.ids)
        return dict(epoch=self.epoch,base_indices=self.base,actual_indices=self.actual,
            base_ids=original,actual_ids=actual,batches=self.read_rows,
            id_multiplicities_equal=True,all_ids_once=True,task_slots_unchanged=True,
            current_and_pinned_unchanged=True,actual_loader_verified=True)


class PlannedBatchSampler:
    def __init__(self,plan):self.plan=plan
    def __len__(self):return len(self.plan)
    def __iter__(self):
        p=self.plan
        while p.cursor<len(p.actual):
            start=p.cursor;end=min(start+p.batch_size,len(p.actual))
            indices=p.actual[start:end];p.last_batch=(start,end);p.cursor=end
            yield indices


class CoverageOrder:
    def __init__(self,policy,seed,task,old,current,load,fingerprint,config=None):
        self.policy=getattr(policy,'module',policy);self.task=task
        self.old={int(k):tuple(v) for k,v in sorted(old.items())}
        self.current=tuple(current);self.load=load;self.fingerprint=fingerprint
        self.config=config or OrderConfig();self.rng=random.Random(seed)
        self.last_step=None;self.events=[]
        self.totals=dict(schedules=0,candidate_gradients=0,committed_tasks=0,
            participating_positions=0,moved_positions=0,cross_batch_moved=0,
            scoring_seconds=0.,management_seconds=0.,io_seconds_nested=0.,fallbacks={})

    def should_schedule(self,step):
        return self.config.enabled and bool(self.old) and step%500==0 and self.last_step!=step

    def _payloads(self,ids):
        start=time.perf_counter();out=[self.load(i) for i in ids]
        self.totals['io_seconds_nested']+=time.perf_counter()-start
        return out

    def _score(self,candidates,refs,new_ids):
        import torch
        from libero.lifelong.replay_selection import isolated_scoring,head_gradient,unit,protective_score
        cfg=self.config;result={};meta={}
        with isolated_scoring(self.policy):
            h=None
            for ids in refs.values():
                g=head_gradient(self.policy,self._payloads(ids)) if ids else None
                if g is None:return {},dict(reason='INVALID_HISTORICAL_DIRECTION')
                h=g/len(refs) if h is None else h+g/len(refs)
            hu=unit(h,cfg.numerical_epsilon)
            if hu is None:return {},dict(reason='INVALID_HISTORICAL_DIRECTION')
            n=head_gradient(self.policy,self._payloads(new_ids)) if new_ids else None
            nu=unit(n,cfg.numerical_epsilon)
            meta['reason']='OK' if nu is not None else 'HISTORY_ONLY_INVALID_CURRENT_DIRECTION'
            for task,ids in candidates.items():
                rows=[]
                for item in ids:
                    g=head_gradient(self.policy,self._payloads([item]));self.totals['candidate_gradients']+=1
                    s=protective_score(g,h,n,cfg.conflict_coefficient,cfg.numerical_epsilon)
                    gu=unit(g,cfg.numerical_epsilon)
                    rows.append(dict(id=item,score=s,
                        history_alignment=None if gu is None else float(torch.dot(gu,hu).item()),
                        current_alignment=None if gu is None or nu is None else float(torch.dot(gu,nu).item()),
                        conflict_penalty=None if gu is None else (0. if nu is None else float(torch.clamp(-torch.dot(gu,nu),min=0).item()))))
                result[task]=rows
        return result,meta

    def schedule(self,plan,step):
        if not self.should_schedule(step):return None
        start=time.perf_counter();self.last_step=step;self.totals['schedules']+=1
        positions,end=plan.candidates(step,self.rng,self.config.max_candidates_per_old_task)
        record=dict(step=step,epoch=plan.epoch,start=plan.cursor,end=end,tasks={},references={})
        candidates={}
        for task,ps in positions.items():
            ids=[plan.ids[plan.actual[p]] for p in ps]
            why='TOO_FEW_POSITIONS' if len(ps)<2 else ('SINGLE_UPDATE_ONLY' if len({p//plan.batch_size for p in ps})<2 else None)
            record['tasks'][str(task)]=dict(positions=ps,candidates=ids,reason=why,committed=False)
            if why is None:candidates[task]=ids
        if candidates:
            for task,ids in self.old.items():
                allowed=[i for i in ids if i not in set(record['tasks'][str(task)]['candidates'])]
                record['references'][task]=self.rng.sample(allowed,min(len(allowed),self.config.reference_per_old_task))
            new_ids=self.rng.sample(self.current,min(len(self.current),self.config.current_reference_size))
            record['current_reference']=new_ids
            scoring=time.perf_counter();scores,meta=self._score(candidates,record['references'],new_ids)
            duration=time.perf_counter()-scoring;self.totals['scoring_seconds']+=duration
            record.update(meta);record['scoring_seconds']=duration
            for task,ids in candidates.items():
                row=record['tasks'][str(task)];details=scores.get(task,[]);row['score_components']=details
                values=[x['score'] for x in details]
                if not details:row['reason']='INVALID_HISTORICAL_DIRECTION'
                elif any(v is None or not math.isfinite(v) for v in values):row['reason']='INVALID_CANDIDATE_GRADIENT'
                elif max(values)-min(values)<=self.config.numerical_epsilon:row['reason']='INDISTINGUISHABLE_SCORES'
                else:
                    ps=row['positions'];before=[plan.actual[p] for p in ps]
                    ordered,permutation=weighted_permutation(before,values,self.rng,self.config.softmax_temperature)
                    ok,why=plan.commit(ps,ordered,end);row.update(committed=ok,reason='OK' if ok else 'CONSTRAINT_ROLLBACK',constraint_error=why)
                    row['permutation']=permutation
                    if ok:
                        row['ordered_ids']=[plan.ids[i] for i in ordered]
                        original_position={i:p for i,p in zip(before,ps)}
                        row['movements']=[dict(id=plan.ids[i],from_position=original_position[i],to_position=p,
                            steps_earlier=original_position[i]//plan.batch_size-p//plan.batch_size) for i,p in zip(ordered,ps)]
                        self.totals['committed_tasks']+=1;self.totals['participating_positions']+=len(ps)
                        self.totals['moved_positions']+=sum(i!=j for i,j in zip(before,ordered))
                        self.totals['cross_batch_moved']+=sum(m['steps_earlier']!=0 for m in row['movements'])
        else:record['reason']='NO_ELIGIBLE_TASK'
        for row in record['tasks'].values():
            if not row['committed']:
                reason=row['reason'];self.totals['fallbacks'][reason]=self.totals['fallbacks'].get(reason,0)+1
        plan.verify();self.events.append(record)
        self.totals['management_seconds']+=time.perf_counter()-start-record.get('scoring_seconds',0.)
        return record

    def state_dict(self):
        return copy.deepcopy(dict(version=1,config=asdict(self.config),task=self.task,
            fingerprint=self.fingerprint,rng=self.rng.getstate(),last_step=self.last_step,totals=self.totals))

    def load_state_dict(self,value):
        assert value['version']==1 and value['config']==asdict(self.config)
        assert value['task']==self.task and value['fingerprint']==self.fingerprint
        self.rng.setstate(value['rng']);self.last_step=value['last_step'];self.totals=copy.deepcopy(value['totals']);self.events=[]
