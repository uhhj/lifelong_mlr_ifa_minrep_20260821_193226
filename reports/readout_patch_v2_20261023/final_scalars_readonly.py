"""Complete private-checkpoint/selection/partition audit and scalar-only export; CPU no forward."""
import os,sys,json,math
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/READOUT_PATCH_OBJECT41_20261023_v2')
os.environ.update(PBD_ROOT=str(Q),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');sys.path.insert(0,str(Q))
from pbd.common import *
from pbd.models import build
from physical_effect_probe.train_probe import scalar_metrics
contract=verify();assert (Q/'controller.exit').read_text().strip()=='0'
assert all(p.read_text().strip()=='0' for p in Q.glob('*.exit'))
for pp in Path('/proc').iterdir():
 if pp.name.isdigit() and int(pp.name)!=os.getpid():
  try:ee=(pp/'environ').read_bytes().split(b'\0')
  except OSError:continue
  assert ('PBD_ROOT='+str(Q)).encode() not in ee,'Own process remains'
gate=js(OUT/'A_GATE.json');phase='SPATIAL' if (OUT/'SPATIAL_GATE.json').exists() else 'READOUT';res=js(OUT/(phase+'_RESULTS.json'))
assert sha(OUT/(phase+'_RESULTS.json'))==gate['source_results_sha256']
kinds=['P0','M','C','Q']+(['G','S'] if phase=='SPATIAL' else []);split=js(OLD/'output/DEMO_SPLITS.json');epochs={};fits=[];cases={};cps={}
for line in (OUT/'EVENTS.jsonl').read_text().splitlines():
 e=json.loads(line)
 if e['event']=='epoch':epochs.setdefault(e['key'],[]).append(e)
for pn in (['READOUT','SPATIAL'] if phase=='SPATIAL' else ['READOUT']):
 cost=js(OUT/(pn+'_COST.json'));assert cost['status']=='COMPLETE';fits+=cost['fits']
assert len(fits)==len(epochs)==len(kinds)*10
def compare(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:compare(a[k],b[k])
 elif isinstance(a,(float,int)) and a is not None:assert np.isclose(a,b,rtol=1e-5,atol=1e-4),(a,b)
 else:assert a==b
for task in (6,7):
 d,x=data(task);del x;meta=d['metadata'];valid=d['mask'][:,0]
 for kind in kinds:
  ii=[];pred={m:[] for m in ('normal','latest_visual_previous','other_demo_visual')}
  for fold in split[str(task)]:
   key='%s_task%d_fold%d'%(kind,task,fold['fold']);es=epochs[key];assert [e['epoch'] for e in es]==list(range(50));best=min(es,key=lambda e:e['validation_position_m'])
   row=next(r for r in fits if r['key']==key);nt=sum(bool(valid[j]) and r['demo_id'] in fold['train'] for j,r in enumerate(meta));assert row['updates']==es[-1]['updates']==math.ceil(nt/128)*50
   case=js(OUT/'cases'/(key+'_b50.json'));sel=case['selection'];cp=Path(sel['path']);assert sha(cp)==sel['sha256'];st=torch.load(cp,map_location='cpu');assert st['epoch']==best['epoch']==sel['selected_epoch'] and st['fold']==fold and st['contract_sha256']==contract
   model=build(kind);model.load_state_dict(st['model'],strict=True);assert all(torch.isfinite(v).all() for v in st['model'].values());cps[str(cp)]=sha(cp);del model,st
   pf=OUT/'private/predictions'/(key+'_b50.pth');assert sha(pf)==case['predictions_sha256'];pp=torch.load(pf,map_location='cpu')
   for part in ('train','validation','test'):
    ids=pp[part]['ids'];expected=[j for j,r in enumerate(meta) if valid[j] and r['demo_id'] in fold[part]];assert ids.tolist()==expected
    for mode,p in pp[part]['predictions'].items():compare(scalar_metrics(p,d['q'][ids],d['previous_q'][ids],meta,ids,d['dynamic'][ids]),case['partitions'][part][mode])
   ii.append(pp['test']['ids'])
   for mode in pred:pred[mode].append(pp['test']['predictions'][mode])
  ids=torch.cat(ii);assert len(ids)==int(valid.sum()) and len(set(ids.tolist()))==len(ids);y=d['q'][ids];yp=d['previous_q'][ids];modes={}
  for mode,ps in pred.items():
   p=torch.cat(ps);er=torch.stack([(p[:,:3]-y[:,:3]).norm(dim=-1),(p[:,3:6]-y[:,3:6]).norm(dim=-1),(p[:,6]-y[:,6]).abs()],-1)*1000
   prev=torch.stack([(p[:,:3]-yp[:,:3]).norm(dim=-1),(p[:,3:6]-yp[:,3:6]).norm(dim=-1)],-1)*1000
   modes[mode]=dict(error_mm=er.tolist(),previous_target_error_mm=prev.tolist())
  cases['%s_task%d'%(kind,task)]=dict(task=task,model=kind,ids=ids.tolist(),demo_ids=[meta[int(i)]['demo_id'] for i in ids],dynamic=d['dynamic'][ids].tolist(),modes=modes)
assert verify()==contract
print(json.dumps(dict(status='PASS_COMPLETE_CP_SELECTION_PARTITIONS_SCALAR_EXPORT',phase=phase,cases=cases,checkpoint_sha256=cps,actual_cost= dict(fits=len(fits),epochs=sum(r['epochs'] for r in fits),updates=sum(r['updates'] for r in fits),fits_detail=fits),selected=gate['selected'],source_results_sha256=sha(OUT/(phase+'_RESULTS.json')),contract_sha256=contract,optimizer_calls=0,model_forwards=0,environment_steps=0,server_writes=0)))
