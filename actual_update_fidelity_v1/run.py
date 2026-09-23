import argparse,contextlib,gc,io,os,time,traceback
import torch
from gradient_relation_experiment import kernel as g
from gradient_relation_experiment.prepare import epoch_indices
from .common import *
from . import engine

def build(start):
 spec=start['plan'];assert sha(ROOT/spec['path'])==spec['sha256'];plan=read(ROOT/spec['path'])
 ds=g.CachedWindows(plan['immutable_feature_origins'],plan['record_sha256'])
 cfg=g.train_config(start['seed'],SCRATCH/'bootstrap');g.bootstrap_train(cfg);algo=g.build_train(cfg)
 sampler=torch.utils.data.DistributedSampler(ds,**{k:plan['sampler'][k] for k in ('seed','num_replicas','rank','shuffle','drop_last')})
 dl=torch.utils.data.DataLoader(ds,batch_size=10,sampler=sampler,num_workers=0)
 with contextlib.redirect_stdout(io.StringIO()):algo.start_task(-1,dl)
 assert algo.loss_scale==1. and g.module(algo).policy_head.loss_coef==1. and not g.module(algo).policy_head.low_eval_noise
 return algo,ds,plan

def load_anchor(algo,start):
 a=start['anchor'];assert sha(a['anchor']['path'])==a['anchor']['sha256'];s=torch.load(a['anchor']['path'],map_location='cpu');g.restore(algo,s)
 checks={k:g.digest(g.capture(algo)[k])==v for k,v in a['witness'].items()};assert all(checks.values()),checks
 return a['stage_step']

def witness(algo,start):
 s=g.capture(algo);checks={k:g.digest(s[k])==v for k,v in start['expected_witness'].items()}
 emit(OUT/'RECOVERY_AUDIT.jsonl',dict(start_id=start['start_id'],checks=checks))
 assert all(checks.values()),('HISTORICAL_RECONSTRUCTION_MISMATCH',start['start_id'],checks)
 assert s['optimizer_step']==start['optimizer_step'] and s['scheduler_step']==start['scheduler_step']
 return s

def hvp_ids(starts):
 groups={}
 for s in starts:groups.setdefault((s['seed'],s['transition']),[]).append(s)
 chosen=[]
 for key,group in sorted(groups.items()):
  for stage in ('30-69','70-99'):chosen.append(next(s['start_id'] for s in group if s['stage']==stage))
  chosen.append([s['start_id'] for s in group if s['stage']=='30-69'][-1])
 for key in sorted(groups)[:2]:chosen.append(groups[key][-1]['start_id'])
 assert len(set(chosen))==20;return chosen

def preflight(starts):
 records=[]
 for seed in (45,46,47):
  for task in (7,8):
   s=next(x for x in starts if x['seed']==seed and x['task_id']==task)
   algo,ds,plan=build(s);step=load_anchor(algo,s);assert step==s['stage_step'];witness(algo,s)
   record=engine.parity(algo,ds,s);records.append(record);write(OUT/'ADAMW_PROPOSAL_PARITY.json',dict(status='RUNNING',records=records))
   print('PARITY',s['start_id'],record['status'],flush=True);del algo,ds;gc.collect();torch.cuda.empty_cache()
 write(OUT/'ADAMW_PROPOSAL_PARITY.json',dict(status='PASS',records=records,criterion='bitwise complete state equivalence plus counters; native observe is independent reference',formal_training_resumed=False))

def measure_all(starts,probes):
 assert read(OUT/'ADAMW_PROPOSAL_PARITY.json')['status']=='PASS'
 hvps=set(hvp_ids(starts));write(OUT/'HVP_SELECTION.json',dict(start_ids=sorted(hvps),trigger=read(OUT/'PROTOCOL.json')['hvp_trigger']))
 for seed in (45,46,47):
  for task in (7,8):
   group=[s for s in starts if s['seed']==seed and s['task_id']==task];todo=[s for s in group if not (OUT/'starts'/s['start_id']/'COMPLETE.json').exists()]
   if not todo:continue
   algo,ds,plan=build(todo[0]);position=None;cache={};rolling=SCRATCH/f'rolling_s{seed}_t{task}.pt'
   for start in todo:
    dest=OUT/'starts'/start['start_id'];dest.mkdir(parents=True,exist_ok=True)
    anchorstep=start['anchor']['stage_step']
    if position is None or anchorstep>position:position=load_anchor(algo,start)
    # Optional isolated rolling state only; no writes to any parent recovery directory.
    if rolling.exists():
     saved=torch.load(rolling,map_location='cpu')
     if saved['manifest_sha256']==sha(OUT/'START_MANIFEST.json') and position<saved['position']<=start['stage_step']:
      g.restore(algo,saved['state']);position=saved['position']
     del saved
    begin=time.monotonic();initial_position=position
    while position<start['stage_step']:
     epoch,bi=divmod(position,plan['batches_per_epoch'])
     if epoch not in cache:cache[epoch]=epoch_indices(plan,epoch)
     g.native(algo,g.batch(ds,cache[epoch][bi*10:(bi+1)*10]),task,epoch)
     previous=position;position+=1;g.ensure_next_epoch_rng(algo,ds,plan,previous,position)
     if position%100==0:
      write(OUT/'STATUS.json',dict(phase='RECONSTRUCT_ORIGIN',seed=seed,task=task,stage_step=position,target=start['start_id'],target_stage_step=start['stage_step'],completed=len(list((OUT/'starts').glob('*/COMPLETE.json'))),seconds=time.monotonic()-begin,steps_this_segment=position-initial_position,pid=os.getpid(),formal_training_resumed=False))
     if position%plan['batches_per_epoch']==0:
      g.atomic_save(rolling,dict(state=g.capture(algo),position=position,manifest_sha256=sha(OUT/'START_MANIFEST.json')))
    base=witness(algo,start)
    if start['start_id'] in hvps:
     hp=SCRATCH/'hvp'/f'{start["start_id"]}.pt'
     if not hp.exists():g.atomic_save(hp,dict(state=base,start_id=start['start_id'],manifest_sha256=sha(OUT/'START_MANIFEST.json')))
    indices=epoch_indices(plan,start['epoch'])[start['batch_index']*10:(start['batch_index']+1)*10];assert indices==start['record_indices']
    write(OUT/'STATUS.json',dict(phase='MEASURE',start_id=start['start_id'],completed=len(list((OUT/'starts').glob('*/COMPLETE.json'))),pid=os.getpid(),formal_training_resumed=False))
    rows,info=engine.measure(algo,ds,start,probes[start['start_id']]);table(dest/'measurements.csv',rows);write(dest/'COMPLETE.json',info)
    g.atomic_save(rolling,dict(state=g.capture(algo),position=position,manifest_sha256=sha(OUT/'START_MANIFEST.json')))
    print('MEASURED',start['start_id'],info['seconds'],'reconstruction',position-initial_position,flush=True)
    del base;gc.collect()
   del algo,ds;gc.collect();torch.cuda.empty_cache()

def hvp(starts,probes):
 from .analysis import summarize
 decision=summarize(partial=False)
 if not decision['hvp_trigger']:return
 selection=read(OUT/'HVP_SELECTION.json')['start_ids'];allrows=[]
 for seed in (45,46,47):
  for task in (7,8):
   group=[s for s in starts if s['seed']==seed and s['task_id']==task and s['start_id'] in selection]
   algo,ds,plan=build(group[0])
   for s in group:
    output=OUT/'hvp'/f'{s["start_id"]}.json'
    if output.exists():allrows.extend(read(output)['rows']);continue
    saved=torch.load(SCRATCH/'hvp'/f'{s["start_id"]}.pt',map_location='cpu');assert saved['manifest_sha256']==sha(OUT/'START_MANIFEST.json');g.restore(algo,saved['state']);witness(algo,s)
    selected=[]
    for t in range(6,task):selected.extend([r for r in probes[s['start_id']] if r['task_id']==t][:3])
    begin=time.monotonic();rows=engine.second_order(algo,ds,s,selected);write(output,dict(rows=rows,seconds=time.monotonic()-begin));allrows.extend(rows)
   del algo,ds;gc.collect();torch.cuda.empty_cache()
 table(OUT/'SECOND_ORDER_DIAGNOSTIC.csv',allrows)
 from .analysis import second_report
 second_report(allrows)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['preflight','all','measure'],default='all');args=parser.parse_args()
 SCRATCH.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.set_num_interop_threads(1)
 starts=read(OUT/'START_MANIFEST.json')['starts'];probes={s['start_id']:[] for s in starts}
 for r in read(OUT/'PROBE_MANIFEST.json')['windows']:probes[r['start_id']].append(r)
 g.init_ddp(SCRATCH/f'ddp_{os.getpid()}')
 try:
  if not (OUT/'ADAMW_PROPOSAL_PARITY.json').exists() or read(OUT/'ADAMW_PROPOSAL_PARITY.json')['status']!='PASS':preflight(starts)
  if args.phase=='preflight':return
  measure_all(starts,probes);hvp(starts,probes)
  from .analysis import summarize
  summarize(partial=False);write(OUT/'STATUS.json',dict(phase='COMPLETE',completed=120,formal_training_resumed=False,pid=os.getpid()))
 except Exception:
  write(OUT/'FAILED.json',dict(traceback=traceback.format_exc(),formal_training_resumed=False,pid=os.getpid()));raise
 finally:
  if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
if __name__=='__main__':main()
