import argparse,json
from pathlib import Path
import torch
from coreset_exp.common import O,write,save,sha,verify
from .science import derive_task,graph
def main():
 ap=argparse.ArgumentParser();ap.add_argument('task',type=int);ap.add_argument('--retain',action='store_true');args=ap.parse_args();t=args.task;verify()
 root=O/'effects/current'/('task%d'%t)
 if args.retain:
  labels=torch.load(root/'labels.pth',map_location='cpu');m=json.loads((O/'ra_coreset/storage'/('task%d.json'%t)).read_text())
  kept={x['destination']:labels[Path(x['source']).name] for x in m['selected']}
  dst=O/'effects/memory'/('task%d.pth'%t)
  if dst.exists():assert sha(dst)==json.loads(dst.with_suffix('.json').read_text())['sha256'];return
  save(dst,kept);write(dst.with_suffix('.json'),dict(sha256=sha(dst),count=len(kept),storage_sha256=sha(O/'ra_coreset/storage'/('task%d.json'%t))))
  return
 assert not root.exists()
 rows,_,proof=derive_task(t);labels={};binding=json.loads((O/'bindings'/('task%d.json'%t)).read_text())['windows']
 for name,w in binding.items():
  r=rows.get(name)
  labels[name]=dict(bc_valid=r is not None,valid=bool(r and r['valid']),sample_sha256=w['sha256'],task=t,sample=name,
   effect=torch.tensor(r['effect'],dtype=torch.float32) if r else torch.zeros(7),next_action=torch.tensor(r['next_action'],dtype=torch.float32) if r else torch.zeros(7),
   group=r['group'] if r else 'INVALID',demo=w['demo_id'] if w['identity_unique'] else 'INVALID',u=w['target'] if w['identity_unique'] else -1,
   grip_command_sign=r['grip_command_sign'] if r else None,identity_unique=w['identity_unique'],full_window_valid=r is not None,next_action_index=(r['u']+1) if r else None,effect_from_state=(r['u']+1) if r and r['valid'] else None,effect_to_state=(r['u']+2) if r and r['valid'] else None,holdout=r['holdout'] if r else False,relative=r['relative'] if r else None,aperture=r['aperture'] if r else None)
 save(root/'labels.pth',labels);write(root/'AUDIT.json',dict(proof,status='PASS',labels_sha256=sha(root/'labels.pth')))
 old=[]
 for j in range(6,t):
  p=O/'effects/memory'/('task%d.pth'%j);assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256']
  old.extend(r for r in torch.load(p,map_location='cpu').values() if r['valid'] and not r['holdout'])
 g=graph([r for r in rows.values() if r['valid'] and not r['holdout']],old)
 expected=json.loads((O/'GATE1_EVIDENCE.json').read_text())['stages'][str(t)]
 assert json.loads(json.dumps(g))==expected,('Offline/actual graph mismatch',t)
 write(root/'SUPPORT.json',g);verify()
if __name__=='__main__':main()
