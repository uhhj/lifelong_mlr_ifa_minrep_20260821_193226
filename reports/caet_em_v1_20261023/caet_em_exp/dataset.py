import shutil
from torch.utils.data import Dataset
from .common import *
class EvidenceDataset(Dataset):
 def __init__(self,paths,rows):self.file_paths=paths;self.rows=rows;assert len(paths)==len(rows)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  x=torch.load(self.file_paths[i],map_location='cpu')['feature'].squeeze(0);r=self.rows[i]
  z={k:r[k] for k in ('action','response','visual','action_mask','response_mask','visual_mask','support')}
  z.update(feature=x,task=r['task'],index=i)
  z['effect']=r.get('effect',torch.zeros(4,64));z['effect_mask']=r.get('effect_mask',torch.zeros(4,dtype=torch.bool))
  return z
def current(task):
 out=ROOT/'evidence'/('task%d'%task);m=js(out/'TEMPORAL_BINDING_AUDIT.json');assert sha(out/'labels.pth')==m['labels_sha256']
 rows=torch.load(out/'labels.pth',map_location='cpu');return rows
def make(arm,task):
 from libero.lifelong.algos.mlr import build_dataset_for_task
 root=arm_root(arm);native=build_dataset_for_task(str(PARENT/'features'),task,str(root/'buffer'),41,start=6)
 labels=current(task);past={}
 for j in range(6,task):
  p=root/'sidecars'/('task%d.pth'%j);meta=js(p.with_suffix('.json'));assert sha(p)==meta['sha256'] and meta['codec_sha256']==sha(frozen_codec());past.update(torch.load(p,map_location='cpu'))
 rows=[]
 for p in native.file_paths:
  p=Path(p);r=past[p.name] if p.name.startswith('t') else labels[p.name]
  assert sha(p)==r['sample_sha256'];rows.append(r)
 assert len(rows)==POPS[task];return EvidenceDataset(native.file_paths,rows)
def retain(arm,task,rows):
 root=arm_root(arm);record=js(PARENT/'ra_coreset/storage'/('task%d.json'%task));selected=record['selected'];assert len(selected)==QUOTAS[task]
 memory={};before={p.name:sha(p) for p in (root/'buffer').glob('*.pt')};root.joinpath('buffer').mkdir(exist_ok=True)
 for item in selected:
  src=Path(item['source']);dst=root/'buffer'/item['destination'];assert sha(src)==item['sha256']
  if dst.exists():assert sha(dst)==item['sha256']
  else:
   temp=dst.with_suffix('.copying');shutil.copyfile(src,temp);assert sha(temp)==item['sha256'];temp.replace(dst)
  r=rows[src.name];assert r['sample_sha256']==item['sha256'];memory[dst.name]=r
 p=root/'sidecars'/('task%d.pth'%task);save(p,memory)
 write(p.with_suffix('.json'),dict(sha256=sha(p),codec_sha256=sha(frozen_codec()),projection_sha256=sha(ROOT/'shared/projection.pth'),normalization_sha256=sha(ROOT/'shared/normalization.pth'),members={n:r['sample_sha256'] for n,r in memory.items()}))
 actual={p.name:sha(p) for p in (root/'buffer').glob('*.pt')};expected=dict(before);expected.update({r['destination']:r['sha256'] for r in selected});assert actual==expected
 write(root/'storage'/('task%d.json'%task),dict(status='PASS',selected=selected,parent_storage_sha256=sha(PARENT/'ra_coreset/storage'/('task%d.json'%task)),sidecar_sha256=sha(p),buffer=actual,counts={str(j):sum(n.startswith('t%d__'%j) for n in actual) for j in range(6,task+1)}))
