"""Ten auxiliary epochs, training-only scales and whole-demo admission."""
import argparse,json,time
import torch
from torch.nn import functional as F
from coreset_exp.common import O,SEED,write,save,sha,verify
from coreset_exp.model import cfg
from libero.lifelong.algos.mlr import build_dataset_for_task
from .dataset import EffectDataset
from .model import EffectModel

def main():
 ap=argparse.ArgumentParser();ap.add_argument('task',type=int);t=ap.parse_args().task
 verify();c=cfg('ra_coreset');ds=EffectDataset(build_dataset_for_task(c.feature_dir,t,c.buffer_dir,SEED,start=6),t)
 root=O/'effects/models'/('task%d'%t);assert not root.exists();g=ds.support
 if not g['stage_supported']:
  write(root/'ACCEPTANCE.json',dict(status='FALLBACK_RA',reason='NO_QUALIFIED_GRAPH',policy_updates=0,effect_updates=0));return
 valid=[i for i,r in enumerate(ds.rows) if r['valid']];tr=[i for i in valid if not ds.rows[i]['holdout']];va=[i for i in valid if ds.rows[i]['holdout']]
 assert tr and va
 begin=time.time();torch.manual_seed(820000000+10000*SEED+100*t)
 X=torch.stack([ds.native[i][0]['feature'].squeeze(0) for i in valid]).cuda()
 A=torch.stack([ds.rows[i]['next_action'] for i in valid]).cuda();Y=torch.stack([ds.rows[i]['effect'] for i in valid]).cuda()
 index={i:k for k,i in enumerate(valid)};tr=[index[i] for i in tr];va=[index[i] for i in va]
 center=Y[tr].mean(0);scale=Y[tr].std(0,unbiased=False).clamp_min(1e-6);target=(Y-center)/scale
 identities={(r['task'],r['sample']):index[i] for i,r in enumerate(ds.rows) if i in index}
 edges=[(identities[(g['current_nodes'][i]['task'],g['current_nodes'][i]['sample'])],identities[(g['old_nodes'][j]['task'],g['old_nodes'][j]['sample'])]) for i,j in g['edges']]
 neighbors={}
 for i,j in edges:neighbors.setdefault(i,[]).append(j);neighbors.setdefault(j,[]).append(i)
 model=EffectModel().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 ledger=[];updates=0
 for ep in range(10):
  generator=torch.Generator().manual_seed(820000000+10000*SEED+100*t+ep)
  order=[tr[k] for k in torch.randperm(len(tr),generator=generator).tolist()];losses=[];pairs_count=0;exposures=0
  model.train()
  # Each training point is an anchor once/epoch. Up to one fixed graph neighbor
  # per anchor gives <=64 entries, no heldout points, and auditable pair exposure.
  for start in range(0,len(order),32):
   anchors=order[start:start+32];batch=list(anchors);pairs=[]
   for k,i in enumerate(anchors):
    ns=neighbors.get(i,[])
    if ns:batch.append(ns[ep%len(ns)]);pairs.append((k,len(batch)-1))
   opt.zero_grad();pred,h=model(X[batch],A[batch]);loss=F.smooth_l1_loss(pred,target[batch])
   if pairs:loss=loss+.1*torch.stack([1-(h[i]*h[j]).sum() for i,j in pairs]).mean()
   assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
   updates+=1;pairs_count+=len(pairs);exposures+=len(batch);losses.append(float(loss.detach()))
  ledger.append(dict(epoch=ep,updates=len(losses),loss=sum(losses)/len(losses),anchor_windows=len(tr),total_window_exposures=exposures,graph_pair_exposures=pairs_count))
  write(root/'epochs.json',ledger)
 model.eval()
 with torch.no_grad():
  prediction=model(X[va],A[va])[0];mae=float((prediction-target[va]).abs().mean())
  constant=float(target[va].abs().mean())
  # A fixed cyclic permutation of next actions, same heldout visual/state inputs.
  perm=model(X[va],A[va].roll(1,0))[0];permuted=float((perm-target[va]).abs().mean())
 action=A[va[:min(64,len(va))]].detach().clone().requires_grad_(True)
 pred=model(X[va[:len(action)]],action)[0];grad=torch.autograd.grad(pred.square().mean(),action)[0]
 sensitive=bool(torch.isfinite(grad).all() and float(grad.abs().sum())>0)
 accepted=mae<constant and mae<permuted and sensitive and len(g['heldout_old_demos'])>=3 and g['heldout_supported']>0
 save(root/'final.pth',dict(model=model.cpu().state_dict(),center=center.cpu(),scale=scale.cpu()))
 write(root/'ACCEPTANCE.json',dict(status='PASS' if accepted else 'FALLBACK_RA',reason=None if accepted else 'HELDOUT_ACTION_CONDITIONED_PREDICTION_NOT_VERIFIED',heldout_mae=mae,training_mean_baseline_mae=constant,permuted_action_mae=permuted,action_gradient_finite_nonzero=sensitive,action_gradient_abs_sum=float(grad.abs().sum()),heldout_supported=g['heldout_supported'],heldout_old_demos=g['heldout_old_demos'],effect_updates=updates,epochs=10,policy_updates=0,seconds=time.time()-begin,train_windows=len(tr),holdout_windows=len(va),frozen_predictor_sha256=sha(root/'final.pth'),support_sha256=sha(O/'effects/current'/('task%d'%t)/'SUPPORT.json')))
 verify()

if __name__=='__main__':main()
