"""Complete arm acceptance, exact outcomes; partial matrices are not scored."""
import argparse
from fractions import Fraction
from .common import *
from .model import Policy,strict_policy_restore
def main():
 ap=argparse.ArgumentParser();ap.add_argument('arm',choices=ARMS);a=ap.parse_args();contract=verify();root=arm_root(a.arm);pack=torch.load(frozen_codec(),map_location='cpu');counts={};total=0;load=[]
 for t in range(6,10):
  hist=js(root/'epochs'/('task%d.json'%t));assert len(hist)==100 and [r['epoch'] for r in hist]==list(range(100)) and all(r['batches']==BATCHES[t] and r['population']==POPS[t] for r in hist)
  ds=js(root/'dataset'/('task%d.json'%t));valid=sum(sum(r['action_mask']) for r in ds['rows']);effect=sum(sum(r['effect_mask']) for r in ds['rows'])
  assert all(r['action_valid']==valid and r['action_invalid']==4*POPS[t]-valid and (a.arm=='B0' or r['effect_valid']==effect) for r in hist)
  assert hist[-1]['steps']==sum(BATCHES[j]*100 for j in range(6,t+1));total+=sum(r['batches'] for r in hist)
  p=Policy(a.arm,pack['model'],pack['statistics']);cp=root/'models'/('task%d.pth'%t);keys=strict_policy_restore(p,torch.load(cp,map_location='cpu'),contract,sha(frozen_codec()));load.append(dict(task=t,strict_keys=keys,sha256=sha(cp)));del p
  storage=js(root/'storage'/('task%d.json'%t));parent=js(PARENT/'ra_coreset/storage'/('task%d.json'%t));assert storage['selected']==parent['selected']
  for j in range(6,t+1):
   out=root/'evaluation';tr=[json.loads(l) for l in (out/('trials_stage%d_task%d.jsonl'%(t,j))).read_text().splitlines()];events=[json.loads(l) for l in (out/('attempts_stage%d_task%d.jsonl'%(t,j))).read_text().splitlines()]
   assert len(tr)==20 and [r['initial_state_index'] for r in tr]==list(range(20));assert all(r['seed']==41 and r['stage']==t and r['task']==j and 1<=r['vector_group_steps']<=600 and r['checkpoint_sha256']==sha(cp) for r in tr)
   assert not [r for r in events if r['event']=='technical_exception'];starts=[r for r in events if r['event']=='vector_attempt_start'];ends=[r for r in events if r['event']=='vector_attempt_complete'];assert len(starts)==len(ends)==2
   for k in (0,1):
    assert starts[k]['group']==ends[k]['group']==k and starts[k]['env_seed']==ends[k]['env_seed']==41+k;assert starts[k]['initial_indices']==list(range(10*k,10*k+10));assert ends[k]['successes']==[r['success'] for r in tr[k*10:k*10+10]]
   assert len([r for r in events if r['event']=='environment_closed'])==1
   counts['%d,%d'%(t,j)]=sum(r['success'] for r in tr)
 assert total==105500 and len(counts)==10
 matrix={k:Fraction(v,20) for k,v in counts.items()};auc=sum(sum(matrix['%d,%d'%(i,j)] for i in range(j,10))/Fraction(10-j) for j in range(6,10))/4
 fwt=sum(matrix['%d,%d'%(j,j)] for j in range(6,10))/4;old=sum(matrix['9,%d'%j] for j in range(6,9))/3;final=sum(matrix['9,%d'%j] for j in range(6,10))/4
 # Project NBT: each old task first averages its later-stage drops; then equal task weight.
 nbt=sum(sum(matrix['%d,%d'%(j,j)]-matrix['%d,%d'%(i,j)] for i in range(j+1,10))/Fraction(9-j) for j in range(6,9))/3
 metrics={k:dict(fraction=str(v),percent=float(v*100)) for k,v in dict(AUC=auc,FWT=fwt,NBT=nbt,OldFinal=old,FinalAll=final).items()}
 write(root/'METRICS.json',dict(status='COMPLETE',counts=counts,metrics=metrics,scope='seed41 exploratory; no significance or protection guarantee'))
 write(root/'FINAL_AUDIT.json',dict(status='PASS',contract=contract,codec_sha256=sha(frozen_codec()),main_epochs=400,main_updates=105500,formal_episodes=200,vector_attempts=20,cells=10,strict_checkpoints=load,metrics_sha256=sha(root/'METRICS.json')))
 event('arm_audit_complete',arm=a.arm)
if __name__=='__main__':main()
