import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import argparse,contextlib,copy,gc,io,time,traceback,unittest
import torch
from gradient_relation_experiment import kernel as g
from gradient_relation_experiment.prepare import epoch_indices
from actual_update_fidelity_v1 import run as prior
from .common import *
from . import engine

def build(start):
 prior.SCRATCH=SCRATCH
 return prior.build(start)

def restore_origin(algo,ds,plan,start):
 candidates=[dict(path=start['anchor']['anchor']['path'],position=start['anchor']['stage_step'],kind='anchor',sha256=start['anchor']['anchor']['sha256'])]
 # Known exact saved full states, only direct paths and the dedicated reconstruction state.
 for p in [Path('/root/actual_update_fidelity_v1')/('rolling_s%d_t%d.pt'%(start['seed'],start['task_id'])),SCRATCH/'rolling.pt']:
  if p.exists():
   z=torch.load(p,map_location='cpu');position=z.get('position')
   same=z.get('cell')==[start['seed'],start['task_id']] if p.parent==SCRATCH else z.get('manifest_sha256')==sha(SOURCE/'START_MANIFEST.json')
   if same and position is not None and position<=start['stage_step']:candidates.append(dict(path=str(p),position=position,kind='rolling'))
   del z
 for p in [Path('/root/actual_update_fidelity_v1/hvp')/(start['start_id']+'.pt')]:
  if p.exists():candidates.append(dict(path=str(p),position=start['stage_step'],kind='hvp'))
 chosen=max(candidates,key=lambda x:x['position'])
 if 'sha256' in chosen:assert sha(chosen['path'])==chosen['sha256']
 saved=torch.load(chosen['path'],map_location='cpu');s=saved if chosen['kind']=='anchor' else saved['state'];g.restore(algo,s)
 assert all(g.compare_states(s,g.capture(algo)).values());del saved,s
 position=chosen['position'];initial=position;cache={};begin=time.monotonic()
 while position<start['stage_step']:
  epoch,bi=divmod(position,plan['batches_per_epoch'])
  if epoch not in cache:cache={epoch:epoch_indices(plan,epoch)}
  g.native(algo,g.batch(ds,cache[epoch][bi*10:(bi+1)*10]),start['task_id'],epoch)
  prev=position;position+=1;g.ensure_next_epoch_rng(algo,ds,plan,prev,position)
  if position%200==0:status('RECONSTRUCT_MISSING_NATIVE_ORIGIN',target=start['start_id'],position=position,target_position=start['stage_step'],steps_completed=position-initial,seconds=time.monotonic()-begin)
  if position%plan['batches_per_epoch']==0:
   g.atomic_save(SCRATCH/'rolling.pt',dict(state=g.capture(algo),position=position,cell=[start['seed'],start['task_id']]))
 indices=epoch_indices(plan,start['epoch'])[start['batch_index']*10:(start['batch_index']+1)*10];assert indices==start['record_indices']
 base=g.capture(algo);assert base['optimizer_step']==start['optimizer_step'] and base['scheduler_step']==start['scheduler_step']
 assert all(torch.isfinite(t).all() for t in base['model'].values() if torch.is_tensor(t))
 expected={k:g.digest(base[k])==v for k,v in start['expected_witness'].items()}
 dest=OUT/'starts'/start['start_id'];dest.mkdir(parents=True,exist_ok=True)
 write(dest/'RESTORE.json',dict(source=chosen,source_load_exact=True,source_historical_witness=expected,hardware_policy='record physical GPU change; no cross-GPU bitwise requirement',missing_native_steps=position-initial,seconds=time.monotonic()-begin,formal_training_resumed=False))
 g.atomic_save(SCRATCH/'rolling.pt',dict(state=base,position=position,cell=[start['seed'],start['task_id']]))
 return base
def test():
 from . import tests
 suite=unittest.defaultTestLoader.loadTestsFromModule(tests);log=io.StringIO()
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 (OUT/'tests.log').write_text(log.getvalue())
 write(OUT/'TEST_REPORT.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),formal_training_resumed=False))
 assert result.wasSuccessful(),log.getvalue()

def measure(stage):
 starts=read(OUT/'START_MANIFEST.json')['starts'];data=read(OUT/'DATA_ROLES.json')['starts']
 assert read(OUT/'INVENTORY_COMPLETE.json')['status']=='PASS'
 if stage=='smoke':
  test();selected=[s for s in starts if s['smoke']]
  if (OUT/'SMOKE_COMPLETE.json').exists():return
 else:
  assert read(OUT/'SMOKE_COMPLETE.json')['status']=='PASS'
  selected=[s for s in starts if not (OUT/'starts'/s['start_id']/'COMPLETE.json').exists()]
 g.init_ddp(SCRATCH/('ddp_'+str(os.getpid())))
 bundles=[]
 try:
  for start in selected:
   if (OUT/'starts'/start['start_id']/'COMPLETE.json').exists():continue
   status('BUILD',start_id=start['start_id'],mode=stage)
   algo,ds,plan=build(start);base=restore_origin(algo,ds,plan,start)
   status('PROPOSAL_AND_GEOMETRY',start_id=start['start_id'],mode=stage)
   bundle=engine.prepare(algo,ds,start,data[start['start_id']]['roles'],base)
   if stage=='smoke':
    bundles.append(bundle);write(OUT/'starts'/start['start_id']/'SMOKE_D_CALIBRATION.json',bundle['noise'])
   else:engine.finish(algo,bundle,data[start['start_id']]['roles'],read(OUT/'NUMERICAL_RESOLUTION.json')['tasks'])
   del algo,ds,base;gc.collect();torch.cuda.empty_cache()
  if stage=='smoke':
   tasks={}
   for b in bundles:
    for role,n in b['noise'].items():
     t=str(n['task_id']);tasks[t]=max(tasks.get(t,0.),n['max_paired_precision_difference'])
   taus={t:dict(tau_num=max(1e-7,5*v),tau=max(1e-4,1e-7,5*v),observed_max_pair_difference=v) for t,v in tasks.items()}
   write(OUT/'NUMERICAL_RESOLUTION.json',dict(tasks=taus,calibration='six preselected smoke D only, actual same FP32 candidate; float32 head vs float64 head, same decoder latents',source_points=[b['start']['start_id'] for b in bundles]))
   for i,b in enumerate(bundles):
    start=b['start'];algo,ds,plan=build(start);g.restore(algo,b['base'])
    engine.finish(algo,b,data[start['start_id']]['roles'],taus)
    del algo,ds;bundles[i]=None;gc.collect();torch.cuda.empty_cache()
   write(OUT/'SMOKE_COMPLETE.json',dict(status='PASS',points=[s['start_id'] for s in selected],decision='technical only; no scientific outcome gate',formal_training_resumed=False))
  status(stage.upper()+'_COMPLETE',completed=len(list((OUT/'starts').glob('*/COMPLETE.json'))))
 finally:
  if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['inventory','smoke','offline','behavior','report']);p.add_argument('--resume',action='store_true');args=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);SCRATCH.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 try:
  if args.stage=='inventory':
   from .inventory import main as run
   run()
  elif args.stage in ['smoke','offline']:measure(args.stage)
  elif args.stage=='report':
   from .report import main as report
   report()
  else:
   from .behavior import main as behavior
   behavior()
 except Exception:
  status('FAILED_STOP',requested_stage=args.stage,traceback=traceback.format_exc());raise
if __name__=='__main__':main()
