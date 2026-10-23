"""Offline diagnostics on current demos and legal old RA, no environments."""
import argparse,time
from torch.utils.data import DataLoader
from .common import *
from .dataset import diagnostic_dataset
from .train import load_codec,targets
from .model import Policy,selected,grouped_huber,huber,masked_mean,strict_policy_restore
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('arm',choices=ARMS);ap.add_argument('task',type=int,choices=range(6,10));a=ap.parse_args();contract=verify();root=arm_root(a.arm);pack,codec=load_codec();p=Policy(a.arm,pack['model'],pack['statistics']).cuda().eval();cp=root/'models'/('task%d.pth'%a.task);strict_policy_restore(p,torch.load(cp,map_location='cpu'),contract,sha(frozen_codec()))
 ds=diagnostic_dataset(a.arm,a.task);targets(ds,codec);records=[];windows=0;tic=time.time()
 for b in DataLoader(ds,32,shuffle=False):
  b=to_device(b);q,z,_=p(b['feature']);base=selected(q);values=dict(predicted_action=grouped_huber(base[0],b['action']))
  if a.arm!='B0':
   gt=selected(p.decoder(b['effect'],b['feature']));values['GT_effect_action']=grouped_huber(gt[0],b['action']);lr,lv=codec.reconstruction(b['effect'],b);values['GT_response']=lr;values['GT_visual']=lv
   lr,lv=codec.reconstruction(z*p.effect_std+p.effect_mean,b);values['predicted_response']=lr;values['predicted_visual']=lv
   for name,altered in [('zero',torch.zeros_like(z)),('permuted',z.roll(1,0))]:
    test=selected(p.decode_z(altered,b['feature']));values[name+'_action_delta']=grouped_huber(test[0],base[0]);values[name+'_mode_change']=(test[2]!=base[2]).float()
   for branch in ('action','response','visual'):
    pert=dict(b);pert[branch]=b[branch].roll(1,0);pert[branch+'_mask']=b[branch+'_mask'].roll(1,0);ep=codec(pert);qp=selected(p.decoder(ep,b['feature']));values['evidence_'+branch+'_permuted_action_error']=grouped_huber(qp[0],b['action'])
  for j,i in enumerate(b['index'].tolist()):
   r=ds.rows[i];entry=dict(sample=r['sample'],task=r['task'],demo=r['demo'],u=r['u'],sample_sha256=r['sample_sha256'],current=r['task']==a.task,metrics={})
   for k,v in values.items():
    mask=b['response_mask'][j] if 'response' in k else (b['visual_mask'][j] if k.endswith('visual') else b['action_mask'][j]);entry['metrics'][k]=dict(sum=float(v[j][mask].sum()),count=int(mask.sum()))
   records.append(entry)
  windows+=len(b['feature'])
 write(root/'diagnostics'/('task%d.json'%a.task),dict(status='PASS_OFFLINE_NOT_CLOSED_LOOP',arm=a.arm,stage=a.task,checkpoint_sha256=sha(cp),codec_sha256=sha(frozen_codec()),windows=windows,policy_forwards=windows,extra_decoder_windows=8*windows if a.arm!='B0' else 0,extra_tokenizer_windows=4*windows,environment_steps=0,optimizer_steps=0,seconds=time.time()-tic,rows=records))
 event('diagnostics_complete',arm=a.arm,task=a.task,windows=windows)
if __name__=='__main__':main()
