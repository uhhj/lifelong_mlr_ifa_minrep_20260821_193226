import argparse,time
from torch.utils.data import DataLoader
from .common import *
from .dataset import EvidenceDataset,current
from .model import Codec,masked_mean,nll,grouped_huber,selected
def loss(model,b):
 e=model(b);q=model.decoder(e,b['feature']);lr,lv=model.reconstruction(e,b)
 bc=nll(q,b['action'],b['action_mask']);ma=masked_mean(grouped_huber(selected(q)[0],b['action']),b['action_mask'])
 rr=masked_mean(lr,b['response_mask']);vv=masked_mean(lv,b['visual_mask']);return bc+.1*ma+.2*rr+.2*vv,dict(nll=float(bc.detach()),mean_huber=float(ma.detach()),response=float(rr.detach()),visual=float(vv.detach()))
def build():
 seed(410003);m=Codec(torch.load(ROOT/'shared/normalization.pth',map_location='cpu')).cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01);return m,opt
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args();contract=verify();capacity();assert js(ROOT/'ADMISSION_CODEC.json')['status']=='PASS'
 rows=current(6);proof=js(ROOT/'evidence/task6/TEMPORAL_BINDING_AUDIT.json');demos=proof['codec_train_demos'];tr=[r for r in rows.values() if r['demo'] in demos and r['action_mask'].any()];va=[r for r in rows.values() if r['demo'] not in demos and r['action_mask'].any()]
 def ds(rs):return EvidenceDataset([PARENT/'features/6'/r['sample'] for r in rs],rs)
 model,opt=build();train=DataLoader(ds(tr),64,shuffle=True,num_workers=0);val=DataLoader(ds(va),64,shuffle=False,num_workers=0)
 path=ROOT/'shared/codec_recovery.pth';start=0;history=[];best=float('inf');selected_epoch=None;best_state=None;steps=0
 if args.resume:
  s=torch.load(path,map_location='cpu');assert s['contract']==contract;restore(model,opt,None,s);start=s['next_epoch'];history=s['history'];best=s['best'];best_state=s['best_state'];selected_epoch=s['selected_epoch'];steps=s['steps']
 else:assert not path.exists()
 for epoch in range(start,100):
  model.train();tic=time.time();trainloss=0
  for b in train:
   b=to_device(b);opt.zero_grad();l,_=loss(model,b);assert torch.isfinite(l);l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),100);opt.step();steps+=1;trainloss+=float(l.detach())
  model.eval();tot={k:0. for k in ('nll','mean_huber','response','visual')};counts={k:0 for k in tot}
  with torch.no_grad():
   for b in val:
    _,parts=loss(model,to_device(b))
    for k,v in parts.items():
     n=int(b[('action' if k in ('nll','mean_huber') else k)+'_mask'].sum());tot[k]+=v*n;counts[k]+=n
  valparts={k:tot[k]/max(counts[k],1) for k in tot};vl=valparts['nll']+.1*valparts['mean_huber']+.2*valparts['response']+.2*valparts['visual'];assert np.isfinite(vl)
  if vl<best:best=vl;selected_epoch=epoch;best_state=cpu(model.state_dict())
  row=dict(epoch=epoch,train_loss=trainloss/len(train),validation_loss=vl,validation_parts=valparts,validation_counts=counts,selected_epoch=selected_epoch,steps=steps,seconds=time.time()-tic);history.append(row)
  save(path,capture(model,opt,None,contract=contract,next_epoch=epoch+1,history=history,best=best,best_state=best_state,selected_epoch=selected_epoch,steps=steps));write(ROOT/'CODEC_TRAINING_AND_VALIDATION.json',dict(status='PARTIAL',train_demos=demos,validation_demos=proof['codec_validation_demos'],epochs=history,optimizer_steps=steps))
  event('codec_epoch',**row)
 model.load_state_dict(best_state,strict=True);model.eval();effects=[]
 with torch.no_grad():
  for b in DataLoader(ds(tr),64,shuffle=False):
   b=to_device(b);valid=b['action_mask']&b['response_mask']&b['visual_mask'];effects.append(model(b)[valid].cpu())
 e=torch.cat(effects);stats=dict(mean=e.mean(0),std=e.std(0,unbiased=False).clamp_min(1e-4))
 frozen=ROOT/'shared/codec_frozen.pth';save(frozen,dict(model=best_state,statistics=stats,selected_epoch=selected_epoch,contract=contract))
 write(ROOT/'CODEC_FREEZE_MANIFEST.json',dict(status='PASS',sha256=sha(frozen),projection_sha256=sha(ROOT/'shared/projection.pth'),normalization_sha256=sha(ROOT/'shared/normalization.pth'),selected_epoch=selected_epoch,validation_loss=best,train_demos=demos,validation_demos=proof['codec_validation_demos'],epochs=100,steps=steps,parameters=sum(p.numel() for p in model.parameters()),environment_steps=0,contract=contract))
 write(ROOT/'CODEC_TRAINING_AND_VALIDATION.json',dict(status='COMPLETE_NOT_PERFORMANCE_GATE',train_demos=demos,validation_demos=proof['codec_validation_demos'],epochs=history,optimizer_steps=steps,selected_epoch=selected_epoch,seconds=sum(r['seconds'] for r in history)))
 event('codec_complete',selected_epoch=selected_epoch,steps=steps)
if __name__=='__main__':main()
