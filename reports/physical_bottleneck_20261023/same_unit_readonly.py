"""Evaluate existing selected native checkpoints, all splits, unchanged labels; no optimizer or writes."""
import os,sys,json,hashlib,subprocess,math
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/PHYSICAL_EFFECT_PROBE_OBJECT41_20261023_engineering03')
os.environ.update(PEP_ROOT=str(Q),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/mlr_ifa/cache/huggingface',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',MUJOCO_GL='egl')
sys.path.insert(0,str(Q))
from physical_effect_probe.common import *
from physical_effect_probe.readout_models import build
import numpy as np
assert (Q/'controller.exit').read_text().strip()=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
contract=verify();O=Q/'output';splits=js(O/'DEMO_SPLITS.json');results=js(O/'STATE_PROBE_RESULTS.json');cases={};batches=0
def stats(e,ids,meta,dynamic):
 ans={}
 for name,mask in [('all',np.ones(len(ids),bool)),('dynamic',dynamic[ids].numpy()),('static',~dynamic[ids].numpy())]:
  z=e[mask];ans[name]=dict(n=len(z),median_mm=np.median(z,axis=0).tolist(),p90_mm=np.quantile(z,.9,axis=0).tolist(),mean_mm=z.mean(0).tolist()) if len(z) else dict(n=0)
 ans['by_demo']={d:dict(n=int(sum(meta[int(i)]['demo_id']==d for i in ids)),median_mm=np.median(e[[meta[int(i)]['demo_id']==d for i in ids]],axis=0).tolist(),p90_mm=np.quantile(e[[meta[int(i)]['demo_id']==d for i in ids]],.9,axis=0).tolist()) for d in sorted({meta[int(i)]['demo_id'] for i in ids})}
 return ans
for task in (6,7):
 data=torch.load(O/'private'/('labels%d.pt'%task),map_location='cpu');x=torch.load(O/'private'/('native%d.pt'%task),map_location='cpu');meta=data['metadata'];valid=data['mask'][:,0]
 assert sha(O/'private'/('native%d.pt'%task))==next(v['sha256'] for v in js(O/'NATIVE_VIEW_MANIFEST.json')['tasks'] if v['task']==task)
 temp=temporal();zz=[]
 with torch.no_grad():
  for start in range(0,len(x),128):
   v=x[start:start+128].cuda();h=temp(v.reshape(len(v),32,512));zz.append(torch.stack([h[:,28],h[:,31]],1).cpu());batches+=1
 emb=torch.cat(zz);del temp;torch.cuda.empty_cache()
 for kind in ('P0','P1','P2','P3','P4'):
  rr=next(r for r in results['results'] if r['task']==task and r['model']==kind and r['train_view']==r['test_view']=='native')
  oldpred=torch.load(O/'private/probes'/('native_native_%s_task%d_predictions.pth'%(kind,task)),map_location='cpu');oldmap={int(i):j for j,i in enumerate(oldpred['ids'])}
  for fold,sel in zip(splits[str(task)],rr['folds']):
   cpfile=O/'private/probes'/sel['key']/'best.pth';assert sha(cpfile)==sel['checkpoint_sha256'];model=build(kind).cuda();model.load_state_dict(torch.load(cpfile,map_location='cpu')['model'],strict=True);model.eval()
   case=dict(task=task,model=kind,fold=fold,checkpoint_sha256=sha(cpfile),selected_epoch=sel['selected_epoch'],partitions={})
   for part in ('train','validation','test'):
    ids=torch.tensor([i for i,r in enumerate(meta) if valid[i] and r['demo_id'] in fold[part]]);pred=[]
    with torch.no_grad():
     for start in range(0,len(ids),128):
      ii=ids[start:start+128];xx=emb[ii,int(kind=='P2')] if kind in ('P1','P2') else x[ii];pred.append((model(xx.cuda())*torch.tensor([.1]*6+[.08],device='cuda')).cpu());batches+=1
    p=torch.cat(pred);y=data['q'][ids];err=torch.stack([(p[:,:3]-y[:,:3]).norm(dim=-1),(p[:,3:6]-y[:,3:6]).norm(dim=-1),(p[:,6]-y[:,6]).abs()],-1).numpy()*1000
    if part=='test':assert torch.allclose(p,oldpred['predictions']['normal'][[oldmap[int(i)] for i in ids]],atol=1e-5,rtol=1e-5)
    case['partitions'][part]=dict(metrics=stats(err,ids,meta,data['dynamic']),axis_absolute_mean_mm=((p-y).abs().mean(0)*1000).tolist(),axis_signed_mean_mm=((p-y).mean(0)*1000).tolist())
   cases[sel['key']]=case;del model
 del x,emb;torch.cuda.empty_cache()
assert verify()==contract
print(json.dumps(dict(status='PASS_SAME_UNIT_EXISTING_CHECKPOINTS',cases=cases,contract_sha256=contract,source_results_sha256=sha(O/'STATE_PROBE_RESULTS.json'),selected_checkpoints=50,model_forward_batches=batches,optimizer_steps=0,environment_episodes=0,server_writes=0,fields=['object_eef','object_goal','aperture'],units='mm',warning='Training/validation repeats across folds; use per-fold comparisons, not independent pooled samples.')))
