"""Causal sidecars; original feature/action payloads are never modified."""
import argparse,json
from pathlib import Path
import torch
from coreset_exp.common import O,write,save,sha,verify
from .science import derive_task,graph

def main():
 ap=argparse.ArgumentParser();ap.add_argument('task',type=int);ap.add_argument('--retain',action='store_true');a=ap.parse_args();t=a.task
 verify();root=O/'effects/current'/('task%d'%t)
 if a.retain:
  labels=torch.load(root/'labels.pth',map_location='cpu');m=json.loads((O/'ra_coreset/storage'/('task%d.json'%t)).read_text())
  kept={x['destination']:labels[Path(x['source']).name] for x in m['selected']}
  dst=O/'effects/memory'/('task%d.pth'%t);assert not dst.exists();save(dst,kept)
  write(dst.with_suffix('.json'),dict(sha256=sha(dst),count=len(kept),storage_sha256=sha(O/'ra_coreset/storage'/('task%d.json'%t))))
  return
 assert not root.exists()
 rows,_,proof=derive_task(t);labels={}
 binding=json.loads((O/'bindings'/('task%d.json'%t)).read_text())['windows']
 for name,w in binding.items():
  r=rows.get(name)
  labels[name]=dict(valid=r is not None,sample_sha256=w['sha256'],task=t,sample=name,
   effect=torch.tensor(r['effect'],dtype=torch.float32) if r else torch.zeros(7),
   next_action=torch.tensor(r['next_action'],dtype=torch.float32) if r else torch.zeros(7),
   group=r['group'] if r else 'INVALID',demo=r['demo'] if r else 'INVALID',u=r['u'] if r else -1,
   holdout=r['holdout'] if r else False,relative=r['relative'] if r else None,aperture=r['aperture'] if r else None)
 save(root/'labels.pth',labels);write(root/'AUDIT.json',dict(proof,status='PASS',labels_sha256=sha(root/'labels.pth')))
 old=[]
 for j in range(6,t):
  p=O/'effects/memory'/('task%d.pth'%j);assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256']
  old.extend(r for r in torch.load(p,map_location='cpu').values() if r['valid'] and not r['holdout'])
 current=[r for r in rows.values() if not r['holdout']]
 g=graph(current,old) if t>6 else dict(current_nodes=[],old_nodes=[],groups={},edges=[],stage_supported=False,supported_current=0)
 if t>6:
  expected=json.loads((O/'GATE1_EVIDENCE.json').read_text())['stages'][str(t)]
  for k in ('current_nodes','old_nodes','groups','edges','stage_supported','supported_current'):
   assert json.loads(json.dumps(g[k]))==expected[k],('Offline/actual support graph mismatch',t,k)
  g['heldout_queries']=expected['heldout_queries'];g['heldout_supported']=expected['heldout_supported'];g['heldout_old_demos']=expected['heldout_old_demos']
 write(root/'SUPPORT.json',g);verify()

if __name__=='__main__':main()
