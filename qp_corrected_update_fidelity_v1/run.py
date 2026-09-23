import argparse,contextlib,gc,io,os,time,traceback
import numpy as np,pandas as pd,torch
from gradient_relation_experiment import kernel as g
from gradient_relation_experiment.prepare import epoch_indices
from actual_update_fidelity_v1 import engine as native
from .common import *
from . import engine

def build(start):
 spec=start['plan'];assert sha(ROOT/spec['path'])==spec['sha256'];plan=read(ROOT/spec['path'])
 ds=g.CachedWindows(plan['immutable_feature_origins'],plan['record_sha256']);cfg=g.train_config(start['seed'],SCRATCH/'bootstrap');g.bootstrap_train(cfg);algo=g.build_train(cfg)
 sampler=torch.utils.data.DistributedSampler(ds,**{k:plan['sampler'][k] for k in ('seed','num_replicas','rank','shuffle','drop_last')});dl=torch.utils.data.DataLoader(ds,batch_size=10,sampler=sampler,num_workers=0)
 with contextlib.redirect_stdout(io.StringIO()):algo.start_task(-1,dl)
 return algo,ds,plan

def anchor(algo,start):
 a=start['anchor'];assert sha(a['anchor']['path'])==a['anchor']['sha256'];saved=torch.load(a['anchor']['path'],map_location='cpu');g.restore(algo,saved)
 actual=g.capture(algo);assert all(g.digest(actual[k])==v for k,v in a['witness'].items())
 return a['stage_step']

def witness(algo,start):
 actual=g.capture(algo);checks={k:g.digest(actual[k])==v for k,v in start['expected_witness'].items()};emit(OUT/'RECOVERY_AUDIT.jsonl',dict(start_id=start['start_id'],checks=checks));assert all(checks.values()),checks

def sanity(starts):
 original=read(SOURCE/'PROBE_MANIFEST.json')['windows'];rows=[]
 for name in read(OUT/'PROTOCOL.json')['parity_starts']:
  s=next(x for x in starts if x['start_id']==name);algo,ds,plan=build(s);assert anchor(algo,s)==s['stage_step'];witness(algo,s)
  parity=native.parity(algo,ds,s);measured,meta=native.measure(algo,ds,s,[r for r in original if r['start_id']==name])
  before=pd.read_csv(SOURCE/'starts'/name/'measurements.csv').sort_values(['arm','alpha','record_id']);after=pd.DataFrame(measured).sort_values(['arm','alpha','record_id'])
  fields=['base_loss','candidate_loss','h_dot_d0','h_dot_direction','d0_norm','predicted_delta_loss','measured_delta_loss','tau_zero'];differences={f:float(np.max(np.abs(before[f].to_numpy()-after[f].to_numpy()))) for f in fields}
  assert all(np.allclose(before[f],after[f],rtol=1e-12,atol=1e-12) for f in fields),differences
  rows.append(dict(start_id=name,status='PASS',proposal_parity=parity,prior_field_max_abs_differences=differences,evidence='same full source witness and exact native proposal implementation; archived scalar/directional/loss witnesses match; previous full d0 vectors were not persisted'))
  write(OUT/'D0_REPRODUCTION_CHECK.json',dict(status='RUNNING',rows=rows));print('D0_PARITY',name,flush=True)
  del algo,ds;gc.collect();torch.cuda.empty_cache()
 write(OUT/'D0_REPRODUCTION_CHECK.json',dict(status='PASS',rows=rows,formal_training_resumed=False))

def iterate(starts,phase,callback):
 for seed in (45,46,47):
  for task in (7,8):
   group=[s for s in starts if s['seed']==seed and s['task_id']==task];todo=[s for s in group if not (OUT/('starts' if phase==1 else 'worksets')/s['start_id']/'COMPLETE.json').exists()]
   if not todo:continue
   algo,ds,plan=build(todo[0]);position=None;cache={};rolling=SCRATCH/f'phase{phase}_rolling.pt'
   for s in todo:
    if position is None or s['anchor']['stage_step']>position:position=anchor(algo,s)
    if rolling.exists():
     saved=torch.load(rolling,map_location='cpu')
     if saved['seed']==seed and saved['task']==task and saved['manifest_sha256']==sha(OUT/'START_MANIFEST.json') and position<saved['position']<=s['stage_step']:g.restore(algo,saved['state']);position=saved['position']
     del saved
    initial=position;begin=time.monotonic()
    def checkpoint():g.atomic_save(rolling,dict(seed=seed,task=task,position=position,state=g.capture(algo),manifest_sha256=sha(OUT/'START_MANIFEST.json')))
    while position<s['stage_step']:
     ep,bi=divmod(position,plan['batches_per_epoch'])
     if ep not in cache:cache[ep]=epoch_indices(plan,ep)
     g.native(algo,g.batch(ds,cache[ep][bi*10:(bi+1)*10]),task,ep);position+=1;g.ensure_next_epoch_rng(algo,ds,plan,position-1,position)
     if position%200==0:write(OUT/'STATUS.json',dict(phase=phase,operation='RECONSTRUCT',start_id=s['start_id'],stage_step=position,target_stage_step=s['stage_step'],pid=os.getpid(),formal_training_resumed=False))
     if position%plan['batches_per_epoch']==0:checkpoint()
    witness(algo,s);assert epoch_indices(plan,s['epoch'])[s['batch_index']*10:(s['batch_index']+1)*10]==s['record_indices']
    write(OUT/'STATUS.json',dict(phase=phase,operation='MEASURE',start_id=s['start_id'],pid=os.getpid(),formal_training_resumed=False))
    callback(algo,ds,s);checkpoint();print('COMPLETE_START',phase,s['start_id'],'replayed',position-initial,'seconds',time.monotonic()-begin,flush=True)
   del algo,ds;gc.collect();torch.cuda.empty_cache()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['preflight','smoke','all'],default='all');args=parser.parse_args();SCRATCH.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);torch.set_num_interop_threads(1);g.init_ddp(SCRATCH/f'ddp_{os.getpid()}')
 starts=read(OUT/'START_MANIFEST.json')['starts']
 try:
  if not (OUT/'D0_REPRODUCTION_CHECK.json').exists() or read(OUT/'D0_REPRODUCTION_CHECK.json')['status']!='PASS':sanity(starts)
  if args.phase=='preflight':return
  refs=read(OUT/'REFERENCE_MANIFEST.json')['cells'];old=read(OUT/'OLD_PROBE_MANIFEST.json')['windows'];new=read(OUT/'NEW_PROBE_MANIFEST.json')['windows']
  def call(algo,ds,s):engine.measure(algo,ds,s,refs[f'{s["seed"]}/{s["task_id"]}'],[r for r in old if r['start_id']==s['start_id']],[r for r in new if r['start_id']==s['start_id']])
  if args.phase=='smoke':
   for name in read(OUT/'PROTOCOL.json')['parity_starts'][:2]:
    s=next(x for x in starts if x['start_id']==name)
    if (OUT/'starts'/name/'COMPLETE.json').exists():continue
    algo,ds,plan=build(s);anchor(algo,s);witness(algo,s);call(algo,ds,s)
    del algo,ds;gc.collect();torch.cuda.empty_cache()
   return
  iterate(starts,1,call)
  from .analysis import summarize
  result=summarize()
  from .audit import run as independent_audit
  independent_audit()
  if result['phase2_gate']:
   from .worksets import run_start,summarize_worksets
   def ws(algo,ds,s):run_start(algo,ds,s,refs[f'{s["seed"]}/{s["task_id"]}'],[r for r in old if r['start_id']==s['start_id']],[r for r in new if r['start_id']==s['start_id']])
   iterate(starts,2,ws);summarize_worksets();result=summarize()
  write(OUT/'STATUS.json',dict(operation='COMPLETE',phase2_executed=result['phase2_gate'],formal_training_resumed=False))
 except Exception:
  write(OUT/'FAILED.json',dict(traceback=traceback.format_exc(),formal_training_resumed=False));raise
 finally:
  if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
if __name__=='__main__':main()
