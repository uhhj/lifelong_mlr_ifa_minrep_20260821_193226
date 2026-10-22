"""Only this method's task-just-learned CP and retained RA members."""
import argparse,json
from pathlib import Path
import torch
from coreset_exp.common import O,sha,save,write,verify
from coreset_exp.model import cfg,strict_downstream
from libero.lifelong.models.brief import BriefPolicy
from .model import params
from .variant import VARIANT
@torch.no_grad()
def main():
 t=int(__import__('sys').argv[1]);verify();assert VARIANT=='EFTR'
 cp=O/'ra_coreset/models'/('mlrifa_%d_model_ep99.pth'%t);mfile=O/'ra_coreset/storage'/('task%d.json'%t);storage=json.loads(mfile.read_text())
 dst=O/'teachers'/('task%d.pth'%t);meta=dst.with_suffix('.json')
 if dst.exists():
  d=json.loads(meta.read_text());assert sha(dst)==d['sha256'] and sha(cp)==d['checkpoint_sha256'];return
 policy=BriefPolicy(cfg('ra_coreset'),None).cuda().eval().requires_grad_(False);assert strict_downstream(policy,cp)==100
 rows={}
 for start in range(0,len(storage['selected']),32):
  batch=storage['selected'][start:start+32];xs=[]
  for b in batch:
   p=O/'ra_coreset/buffer'/b['destination'];assert sha(p)==b['sha256'];xs.append(torch.load(p,map_location='cpu')['feature'].squeeze(0))
  means,std,logits=params(policy,torch.stack(xs).cuda());mode=logits.argmax(-1)
  for i,b in enumerate(batch):rows[b['destination']]=dict(sample_sha256=b['sha256'],means=means[i].cpu(),std=std[i].cpu(),logits=logits[i].cpu(),mode=int(mode[i]))
 save(dst,rows);write(meta,dict(status='PASS',task=t,count=len(rows),sha256=sha(dst),checkpoint_sha256=sha(cp),storage_sha256=sha(mfile),teacher='own_task_ep99',eval_mode=True,gradient=False))
 verify()
if __name__=='__main__':main()
