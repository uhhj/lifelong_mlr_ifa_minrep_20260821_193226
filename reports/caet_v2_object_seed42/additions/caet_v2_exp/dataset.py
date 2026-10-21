import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from coreset_exp.common import O,sha

class EffectDataset(Dataset):
 def __init__(self,native,task):
  self.native=native;self.file_paths=native.file_paths;self.task=task;self.labels={}
  root=O/'effects/current'/('task%d'%task)
  assert sha(root/'labels.pth')==json.loads((root/'AUDIT.json').read_text())['labels_sha256']
  self.labels.update({str(O/'features'/str(task)/n):r for n,r in torch.load(root/'labels.pth',map_location='cpu').items()})
  for j in range(6,task):
   p=O/'effects/memory'/('task%d.pth'%j);assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256']
   self.labels.update({str(O/'ra_coreset/buffer'/n):r for n,r in torch.load(p,map_location='cpu').items()})
  assert all(str(p) in self.labels and sha(p)==self.labels[str(p)]['sample_sha256'] for p in self.file_paths)
  self.support=json.loads((root/'SUPPORT.json').read_text())
  supported={(self.support['current_nodes'][i]['task'],self.support['current_nodes'][i]['sample']) for i,j in self.support['edges']}
  self.rows=[self.labels[str(p)] for p in self.file_paths]
  self.supported=[(r['task'],r['sample']) in supported for r in self.rows]
 def __len__(self):return len(self.native)
 def __getitem__(self,i):
  x,t=self.native[i];r=self.rows[i]
  x.update(effect_valid=r['valid'],effect_target=r['effect'],next_action=r['next_action'],effect_holdout=r['holdout'],transfer_supported=self.supported[i],v2_index=i)
  return x,t
