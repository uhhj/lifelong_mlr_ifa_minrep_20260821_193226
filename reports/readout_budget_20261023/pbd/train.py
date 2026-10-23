"""Finite prospective readout/spatial comparisons. All selection uses validation only."""
import argparse,copy,math
from torch.nn import functional as F
from .common import *
from .models import build
from physical_effect_probe.train_probe import scalar_metrics,donors,perturb

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['budget']);a=ap.parse_args();verify();assert js(OUT/(a.phase.upper()+'_ADMISSION.json'))['status']=='PASS_ISOLATED_GPU'
 split=js(OLD/'output/DEMO_SPLITS.json');kinds=['P0','G','S'];costs=[]
 for task in (6,7):
  d,x=data(task);meta=d['metadata'];valid=d['mask'][:,0];di=donors(meta);x=x.cuda();patch=None;window=None
  if a.phase=='budget':
   pf=V2/'output/private'/('patch%d.pth'%task);assert sha(pf)==next(v['sha256'] for v in js(V2/'output/PATCH_MANIFEST.json')['tasks'] if v['task']==task)
   pp=torch.load(pf,map_location='cpu');patch=pp['patch'].cuda();window=pp['window_indices'];del pp
  for kind in kinds:
   budgets=[50,100,200];micro=16 if kind in ('G','S') else 128
   for fold in split[str(task)]:
    key='%s_task%d_fold%d'%(kind,task,fold['fold']);assert not (OUT/'cases'/(key+'_b50.json')).exists(),'No implicit retry'
    seed(41+task*100+fold['fold']);model=build(kind).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);ids={p:torch.tensor([i for i,r in enumerate(meta) if valid[i] and r['demo_id'] in fold[p]]) for p in ('train','validation','test')}
    def batch(ii,mode='normal'):
     xx=perturb(x[ii],mode,x[di[ii]] if mode=='other_demo_visual' else None)
     if kind!='S':return xx,None
     wi=window[ii].clone()
     if mode=='other_demo_visual':wi=window[di[ii]].clone()
     if mode=='latest_visual_previous':wi[:,-1]=wi[:,-2]
     return xx,patch[wi]
    def forward(ii,mode='normal'):
     xx,pp=batch(ii,mode);return model(xx,pp) if kind in ('G','S') else model(xx)
    def prediction(ii,mode='normal'):
     values=[]
     with torch.no_grad():
      for st in range(0,len(ii),micro):values.append((forward(ii[st:st+micro],mode)*SCALE.cuda()).cpu())
     return torch.cat(values)
    best=float('inf');bestep=None;bestmodel=None;updates=0;selected={};begin=time.time();training_forward_batches=0;validation_forward_batches=0;test_forward_batches=0
    for epoch in range(200):
     model.train();order=ids['train'][torch.randperm(len(ids['train']))];losses=[]
     for start in range(0,len(order),128):
      whole=order[start:start+128];opt.zero_grad(set_to_none=True);total=0.
      for st in range(0,len(whole),micro):
       ii=whole[st:st+micro];loss=F.smooth_l1_loss(forward(ii),d['q'][ii].cuda()/SCALE.cuda());assert torch.isfinite(loss);weighted=loss*(len(ii)/len(whole));weighted.backward();total+=float(weighted.detach());training_forward_batches+=1
      assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step();updates+=1;losses.append(total)
     model.eval();pred=prediction(ids['validation']);validation_forward_batches+=math.ceil(len(ids['validation'])/micro)
     score=float(((pred[:,:3]-d['q'][ids['validation'],:3]).norm(dim=-1)+(pred[:,3:6]-d['q'][ids['validation'],3:6]).norm(dim=-1)).mean()/2)
     if score<best:best=score;bestep=epoch;bestmodel={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
     if epoch+1 in budgets:
      cp=OUT/'private/checkpoints'/(key+'_b%d.pth'%(epoch+1));save(cp,dict(model=bestmodel,kind=kind,task=task,fold=fold,budget=epoch+1,epoch=bestep,validation_score_m=best,contract_sha256=sha(ROOT/'CONTRACT.json')));selected[epoch+1]=dict(path=str(cp),sha256=sha(cp),selected_epoch=bestep,validation_score_m=best)
     save(OUT/'private/recovery.pth',dict(version=1,key=key,epoch=epoch,next_epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),best_model=bestmodel,best_epoch=bestep,best_score=best,selected=selected,updates=updates,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),modes={n:v.training for n,v in model.named_modules()},requires_grad={n:p.requires_grad for n,p in model.named_parameters()},grads={n:None if p.grad is None else p.grad for n,p in model.named_parameters()},contract_sha256=sha(ROOT/'CONTRACT.json'),patch_manifest_sha256=sha(V2/'output/PATCH_MANIFEST.json') if a.phase=='budget' else None))
     log('epoch',key=key,epoch=epoch,updates=updates,training_loss=float(np.mean(losses)),validation_position_m=score,best_epoch=bestep,seconds=time.time()-begin)
     if epoch%10==0:capacity(50*1024**2)
    for budget,sel in selected.items():
     cp=torch.load(sel['path'],map_location='cpu');model.load_state_dict(cp['model'],strict=True);model.eval();case=dict(task=task,kind=kind,fold=fold,budget=budget,selection=sel,partitions={},contract_sha256=sha(ROOT/'CONTRACT.json'));private={}
     for part,ii in ids.items():
      modes=['normal','latest_visual_previous','other_demo_visual'] if part=='test' else ['normal'];case['partitions'][part]={};private[part]=dict(ids=ii,predictions={})
      for mode in modes:
       pred=prediction(ii,mode);test_forward_batches+=math.ceil(len(ii)/micro);private[part]['predictions'][mode]=pred
       case['partitions'][part][mode]=scalar_metrics(pred,d['q'][ii],d['previous_q'][ii],meta,ii,d['dynamic'][ii])
       if budget==50:
        oldpf=V2/'output/private/predictions'/(key+'_b50.pth');oldcase=js(V2/'output/cases'/(key+'_b50.json'));assert sha(oldpf)==oldcase['predictions_sha256']
        op=torch.load(oldpf,map_location='cpu')[part];assert torch.equal(ii,op['ids']);ref=op['predictions'][mode]
        parity=dict(max_abs_prediction_m=float((pred-ref).abs().max()),allclose=bool(torch.allclose(pred,ref,atol=1e-5,rtol=1e-5)),parent_predictions_sha256=sha(oldpf))
        case.setdefault('v2_50_epoch_parity',{}).setdefault(part,{})[mode]=parity;assert parity['allclose'],'Unexpected 50-epoch prefix reproduction mismatch'
     pf=OUT/'private/predictions'/(key+'_b%d.pth'%budget);save(pf,private);case['predictions_sha256']=sha(pf);write(OUT/'cases'/(key+'_b%d.json'%budget),case)
    row=dict(key=key,phase=a.phase,kind=kind,task=task,fold=fold['fold'],epochs=200,updates=updates,training_forward_batches=training_forward_batches,validation_forward_batches=validation_forward_batches,selected_evaluation_forward_batches=test_forward_batches,parameters=sum(p.numel() for p in model.parameters()),microbatch=micro,effective_batch=128,seconds=time.time()-begin,budgets=budgets,selection=selected)
    costs.append(row);write(OUT/(a.phase.upper()+'_COST.json'),dict(status='PARTIAL',fits=costs));log('fit_complete',**row);del model,opt;torch.cuda.empty_cache()
  del x,patch,window;torch.cuda.empty_cache()
 verify();write(OUT/(a.phase.upper()+'_COST.json'),dict(status='COMPLETE',fits=costs));log('phase_complete',phase=a.phase,fits=len(costs))
if __name__=='__main__':main()
