"""100 fixed whole-demo fits. Test outcomes never influence fitting or checkpoint selection."""
import argparse,copy,math,time
import numpy as np
import torch
from torch.nn import functional as F
from .common import *
from .readout_models import build,perturb

SCALE=torch.tensor([.1]*6+[.08])
MODES=('normal','latest_visual_previous','other_demo_visual')

def originals(task,meta):
 return torch.cat([torch.load(PARENT/'features'/str(task)/r['sample'],map_location='cpu')['feature'] for r in meta])

def donors(meta):
 demos=sorted({r['demo_id'] for r in meta});out=[]
 for r in meta:
  dd=demos[(demos.index(r['demo_id'])+1)%len(demos)];c=[i for i,s in enumerate(meta) if s['demo_id']==dd and s['state_valid']]
  out.append(min(c,key=lambda i:(abs(meta[i]['last_observation_index']-r['last_observation_index']),meta[i]['sample'])))
 return torch.tensor(out)

def embeddings(task,views,meta):
 m=temporal();result={};di=donors(meta)
 for view,x in views.items():
  result[view]={}
  for mode in MODES:
   ys=[]
   for st in range(0,len(x),128):
    v=perturb(x[st:st+128].cuda(),mode,x[di[st:st+128]].cuda())
    with torch.no_grad():h=m(v.reshape(len(v),32,512));ys.append(torch.stack([h[:,28],h[:,31]],dim=1).cpu())
   result[view][mode]=torch.cat(ys)
   log('frozen_readout_export',task=task,view=view,mode=mode,windows=len(x))
 del m;torch.cuda.empty_cache();return result

def predict(model,x,ids,kind,mode,meta):
 out=[];di=donors(meta) if mode=='other_demo_visual' else None
 with torch.no_grad():
  for st in range(0,len(ids),128):
   ii=ids[st:st+128]
   if kind in ('P1','P2'):v=x[ii].cuda()
   else:v=perturb(x[ii].cuda(),mode,x[di[ii]].cuda() if di is not None else None)
   out.append((model(v)*SCALE.cuda()).cpu())
 return torch.cat(out)

def scalar_metrics(pred,target,prev,meta,ids,dynamic):
 errors=torch.stack([(pred[:,:3]-target[:,:3]).norm(dim=-1),(pred[:,3:6]-target[:,3:6]).norm(dim=-1),(pred[:,6]-target[:,6]).abs()],-1).numpy()*1000
 previous_errors=torch.stack([(pred[:,:3]-prev[:,:3]).norm(dim=-1),(pred[:,3:6]-prev[:,3:6]).norm(dim=-1)],-1).numpy()*1000
 demos=np.array([meta[int(i)]['demo_id'] for i in ids]);dy=dynamic.numpy();answer={}
 for field,j in [('object_eef',0),('object_goal',1),('aperture',2)]:
  ans={}
  for name,mask in [('all',np.ones(len(ids),bool)),('dynamic',dy),('static',~dy)]:
   e=errors[mask,j];ans[name]={'n':len(e),'median_mm':float(np.median(e)) if len(e) else None,'p90_mm':float(np.quantile(e,.9)) if len(e) else None,'by_demo':{}}
   for d in sorted(set(demos[mask])):
    ii=mask&(demos==d);val=dict(n=int(ii.sum()),median_mm=float(np.median(errors[ii,j])))
    if j<2:val['previous_target_median_mm']=float(np.median(previous_errors[ii,j]))
    ans[name]['by_demo'][d]=val
  answer[field]=ans
 return answer

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tasks',type=int,nargs='+',default=[6,7]);ap.add_argument('--models',nargs='+',default=['P0','P1','P2','P3','P4']);ap.add_argument('--folds',type=int,default=5);ap.add_argument('--seed',type=int,default=41);a=ap.parse_args()
 assert a.tasks==[6,7] and a.models==['P0','P1','P2','P3','P4'] and a.folds==5 and a.seed==41
 verify();assert js(OUT/'VISIBILITY_AUDIT.json')['status']=='PASS';assert not (OUT/'STATE_PROBE_RESULTS.json').exists()
 split=js(OUT/'DEMO_SPLITS.json');results=[];allcost=[];tstart=time.time()
 for task in (6,7):
  data=torch.load(OUT/'private'/('labels%d.pt'%task));meta=data['metadata'];views={'original':originals(task,meta),'native':torch.load(OUT/'private'/('native%d.pt'%task))};emb=embeddings(task,views,meta)
  valid=data['mask'][:,0];allids=torch.arange(len(meta));target=data['q']
  for domain in ('original','native'):
   for kind in a.models:
    heldout={v:{mode:[] for mode in MODES} for v in (('original','native') if domain=='original' else ('native',))};heldids=[];selection=[]
    for fold in split[str(task)]:
     f=fold['fold'];key='%s_%s_task%d_fold%d'%(domain,kind,task,f);seed(41+task*100+f);model=build(kind).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
     train=allids[valid & torch.tensor([r['demo_id'] in fold['train'] for r in meta])];val=allids[valid & torch.tensor([r['demo_id'] in fold['validation'] for r in meta])];test=allids[valid & torch.tensor([r['demo_id'] in fold['test'] for r in meta])]
     assert len(train)>0 and len(val)>0 and len(test)>0
     x=emb[domain]['normal'][:,int(kind=='P2')] if kind in ('P1','P2') else views[domain]
     best=float('inf');bestep=None;updates=0;begin=time.time();rec=OUT/'private/probes'/key
     for epoch in range(50):
      model.train();order=train[torch.randperm(len(train))];losses=[]
      for st in range(0,len(order),128):
       ii=order[st:st+128];opt.zero_grad(set_to_none=True);loss=F.smooth_l1_loss(model(x[ii].cuda()),target[ii].cuda()/SCALE.cuda());assert torch.isfinite(loss)
       loss.backward();opt.step();updates+=1;losses.append(float(loss.detach()))
      model.eval();p=predict(model,x,val,kind,'normal',meta)
      score=float(((p[:,:3]-target[val,:3]).norm(dim=-1)+(p[:,3:6]-target[val,3:6]).norm(dim=-1)).mean()/2)
      if score<best:
       best=score;bestep=epoch;save(rec/'best.pth',dict(model=model.state_dict(),epoch=epoch,validation_score_m=best,kind=kind,fold=fold,domain=domain,contract_sha256=sha(ROOT/'CONTRACT.json')))
      save(OUT/'private/probe_recovery.pth',dict(key=key,epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),updates=updates,best=best,best_epoch=bestep,model_training=model.training))
      log('probe_epoch',key=key,epoch=epoch,updates=updates,training_loss=float(np.mean(losses)),validation_position_m=score)
     cp=torch.load(rec/'best.pth');model.load_state_dict(cp['model'],strict=True);model.eval();heldids.append(test)
     for testview in heldout:
      for mode in MODES:
       xx=emb[testview][mode][:,int(kind=='P2')] if kind in ('P1','P2') else views[testview]
       heldout[testview][mode].append(predict(model,xx,test,kind,mode,meta))
     row=dict(key=key,selected_epoch=bestep,validation_score_m=best,checkpoint_sha256=sha(rec/'best.pth'),updates=updates,epochs=50,seconds=time.time()-begin,train_windows=len(train),validation_windows=len(val),test_windows=len(test),trainable_parameters=sum(p.numel() for p in model.parameters()))
     selection.append(row);allcost.append(row);write(OUT/'PROBE_COST_PARTIAL.json',allcost);log('probe_fit_complete',**row)
    ids=torch.cat(heldids);assert len(ids)==int(valid.sum()) and len(set(ids.tolist()))==len(ids)
    for tv,ms in heldout.items():
     result=dict(task=task,model=kind,train_view=domain,test_view=tv,folds=selection,interventions={})
     private={}
     for mode,preds in ms.items():
      pred=torch.cat(preds);result['interventions'][mode]=scalar_metrics(pred,target[ids],data['previous_q'][ids],meta,ids,data['dynamic'][ids]);private[mode]=pred
     save(OUT/'private/probes'/('%s_%s_%s_task%d_predictions.pth'%(domain,tv,kind,task)),dict(ids=ids,predictions=private))
     results.append(result);write(OUT/'STATE_PROBE_PARTIAL.json',dict(status='PARTIAL',results=results))
    del model,opt
  del views,emb;torch.cuda.empty_cache()
 write(OUT/'STATE_PROBE_RESULTS.json',dict(status='COMPLETE_NOT_GATE_ACCEPTED',results=results,elapsed_seconds=time.time()-tstart,environment_steps=0,policy_updates=0))
 write(OUT/'COST_AUDIT.json',dict(status='A_COMPLETE_B_NOT_STARTED',probe_fits=len(allcost),probe_epochs=sum(r['epochs'] for r in allcost),probe_updates=sum(r['updates'] for r in allcost),policy_updates=0,environment_steps=0,costs=allcost))
 log('probes_complete',fits=len(allcost))
if __name__=='__main__':main()
