"""Record all controls' ideal/realized deletion sizes, without any D/T evaluation."""
import time,copy,gc
import torch
from .common import *

def records(head_keep,x,directions,engagement,rank):
 rows=[];x2=float(x.square().sum());all2=engagement['conditional_increment_norm']**2
 for name,t in directions.items():
  actual=head_keep.double()-(head_keep.double()-t.double()).float().double()
  rows.append(dict(method=name,ideal_delete_norm=float(t.norm()),realized_delete_norm=float(actual.norm()),materialization_error_norm=float((actual-t).norm()),rho_head=float(t.square().sum())/x2 if x2 else None,rho_all=float(t.square().sum())/all2 if all2 else None,random_rank_over_P=rank/len(x),below_physical_resolution=bool(torch.count_nonzero(actual)==0)))
 return rows

def missing_smoke():
 starts=read(OUT/'START_MANIFEST.json')['starts']
 missing=[s for s in starts if not (OUT/'starts'/s['start_id']/'CONTROL_MATERIALIZATION.json').exists()]
 if not missing:return
 assert all(s['smoke'] for s in missing),'Only initial six smoke points may need metadata supplementation'
 from . import run,engine
 from gradient_relation_experiment import kernel as g
 # Separate scratch/audit paths; never overwrite a completed state's RESTORE.
 run.OUT=OUT/'control_metadata_state_audit';run.SCRATCH=SCRATCH/'control_metadata'
 run.SCRATCH.mkdir(parents=True,exist_ok=True);g.init_ddp(run.SCRATCH/('ddp_'+str(os.getpid())))
 try:
  for start in missing:
   begin=time.monotonic();sid=start['start_id'];dest=OUT/'starts'/sid
   status('CONTROL_METADATA_ONLY',start_id=sid,no_GN_rebuild=True,no_D_T_evaluation=True)
   algo,ds,plan=run.build(start);base=run.restore_origin(algo,ds,plan,start)
   states,parity=engine.pair_proposal(algo,ds,start,base)
   saved=torch.load(SCRATCH/(sid+'_behavior.pt'),map_location='cpu')
   for name,v in saved['native_model'].items():
    key=name if name in states['M'] else 'module.'+name
    assert torch.equal(states['M'][key],v),'Native candidate differs from frozen smoke endpoint'
   x=(engine.flat_state(algo,states['M'],True).double()-engine.flat_state(algo,states['R'],True).double()).cuda()
   eng=read(dest/'ENGAGEMENT.json');geo=read(dest/'GEOMETRY.json');phys=read(dest/'PHYSICAL.json')
   assert abs(float(x.norm())-eng['conditional_head_norm'])<=1e-12
   # Controls depend on GN norm, not on the unretained GN basis vector.
   placeholder=torch.zeros_like(x);placeholder[0]=eng['projection_norm']
   dirs=engine.random_controls(x,placeholder,geo['rank'],read(OUT/'PROTOCOL.json')['random_seeds'])
   head=engine.flat_state(algo,states['M'],True).cuda();rows=records(head,x,dirs,eng,geo['rank'])
   rows[0].update(ideal_delete_norm=phys['ideal_delete_norm'],realized_delete_norm=phys['realized_delete_norm'],materialization_error_norm=phys['materialization_error_norm'],rho_head=eng['rho_head'],rho_all=eng['rho_all'],below_physical_resolution=phys['status']=='BELOW_PHYSICAL_RESOLUTION')
   g.restore(algo,base);assert all(g.compare_states(base,g.capture(algo)).values())
   write(dest/'CONTROL_MATERIALIZATION.json',dict(rows=rows,source='Supplementary metadata only; native candidate bitwise verified against frozen smoke endpoint; no repeated GN or D/T measurement',seconds=time.monotonic()-begin,formal_training_resumed=False))
   del algo,ds,base,states,saved,x,dirs,head,placeholder;gc.collect();torch.cuda.empty_cache()
 finally:
  if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1);missing_smoke()
