"""ALIGN labels only: the same next-action mask; effects are unused placeholders."""
import argparse,json,os
from pathlib import Path
import torch
from coreset_exp.common import O,verify,save,write,sha
from .bc_labels import derive
def main():
 ap=argparse.ArgumentParser();ap.add_argument('task',type=int);ap.add_argument('--retain',action='store_true');a=ap.parse_args();verify();t=a.task
 root=O/'effects/current'/('task%d'%t)
 if a.retain:
  labels=torch.load(root/'labels.pth',map_location='cpu');m=json.loads((O/'ra_coreset/storage'/('task%d.json'%t)).read_text())
  kept={x['destination']:labels[Path(x['source']).name] for x in m['selected']};dst=O/'effects/memory'/('task%d.pth'%t)
  if dst.exists():assert sha(dst)==json.loads(dst.with_suffix('.json').read_text())['sha256'];return
  save(dst,kept);write(dst.with_suffix('.json'),dict(sha256=sha(dst),count=len(kept),storage_sha256=sha(O/'ra_coreset/storage'/('task%d.json'%t))));return
 assert not root.exists()
 rows,proof=derive(os.environ['RA_BASE'],t);expected=json.loads((O/'GATE1_EVIDENCE.json').read_text())['tasks'][str(t)]
 assert proof==expected
 binding=json.loads((O/'bindings'/('task%d.json'%t)).read_text())['windows'];labels={}
 for n,w in binding.items():
  r=rows.get(n);unique=w.get('identity_unique',True)
  labels[n]=dict(bc_valid=r is not None,valid=False,sample_sha256=w['sha256'],task=t,sample=n,
    effect=torch.zeros(7),next_action=torch.tensor(r['next_action'],dtype=torch.float32) if r else torch.zeros(7),
    group='ALIGN_UNUSED',demo=w['demo_id'] if unique else 'INVALID',u=w['target'] if unique else -1,
    identity_unique=unique,full_window_valid=r is not None,next_action_index=(r['u']+1) if r else None,
    holdout=False,relative=None,aperture=None)
 save(root/'labels.pth',labels);write(root/'AUDIT.json',dict(proof,labels_sha256=sha(root/'labels.pth')))
 write(root/'SUPPORT.json',dict(status='NOT_APPLICABLE_ALIGN_HAS_NO_EFFECT_OR_CONTRAST',edges=[],current_nodes=[],old_nodes=[]));verify()
if __name__=='__main__':main()
