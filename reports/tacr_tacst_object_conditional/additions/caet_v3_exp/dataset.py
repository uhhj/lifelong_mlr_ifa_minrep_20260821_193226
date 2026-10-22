import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from coreset_exp.common import O,sha
class EffectDataset(Dataset):
 def __init__(self,native,task):
  self.native=native;self.file_paths=native.file_paths;self.task=task;labels={};root=O/'effects/current'/('task%d'%task)
  assert sha(root/'labels.pth')==json.loads((root/'AUDIT.json').read_text())['labels_sha256']
  labels.update({str(O/'features'/str(task)/n):r for n,r in torch.load(root/'labels.pth',map_location='cpu').items()})
  for j in range(6,task):
   p=O/'effects/memory'/('task%d.pth'%j);assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256']
   labels.update({str(O/'ra_coreset/buffer'/n):r for n,r in torch.load(p,map_location='cpu').items()})
  assert all(str(p) in labels and sha(p)==labels[str(p)]['sample_sha256'] for p in self.file_paths)
  self.rows=[labels[str(p)] for p in self.file_paths];self.support=json.loads((root/'SUPPORT.json').read_text());self.positive={};self.negative={}
  lookup={(r['task'],r['sample']):i for i,r in enumerate(self.rows)};g=self.support
  for ci,oi in g['edges']:
   c=g['current_nodes'][ci];o=g['old_nodes'][oi];i=lookup[c['task'],c['sample']];j=lookup[o['task'],o['sample']]
   self.positive.setdefault(i,[]).append(j)
  old=[i for i,r in enumerate(self.rows) if r['task']<task and r['valid'] and not r['holdout'] and '/STATIC/' not in r['group']]
  scale=np.array([.01]*3+[np.pi/18]*3+[.01])
  for i in self.positive:
   self.positive[i]=self.unique(self.positive[i]);r=self.rows[i]
   neg=[j for j in old if self.rows[j]['group'].split('/')[1]!=r['group'].split('/')[1] and np.linalg.norm((np.asarray(self.rows[j]['effect'])-np.asarray(r['effect']))/scale)>=2]
   self.negative[i]=self.unique(neg)
 def unique(self,ids):
  out={}
  for i in sorted(ids,key=lambda i:(self.rows[i]['task'],self.rows[i]['sample'])):
   r=self.rows[i];out.setdefault((r['task'],r['demo'],r['u']),i)
  return list(out.values())
 def __len__(self):return len(self.native)
 def __getitem__(self,i):
  x,t=self.native[i];r=self.rows[i]
  x.update(effect_valid=r['valid'],effect_target=r['effect'],next_action=r['next_action'],causal_bc_valid=r['bc_valid'],v3_index=i)
  return x,t
 def features(self,ids):return torch.stack([self.native[i][0]['feature'].squeeze(0) for i in ids]).cuda()
