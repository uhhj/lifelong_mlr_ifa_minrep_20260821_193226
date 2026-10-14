import json,hashlib,sys
from pathlib import Path
import torch,numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def same(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
out=[];hashes={}
for seed in (42,43,44):
 root=Path(f'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{seed}_20261013')
 for stage in range(6,10):
  for task in range(6,stage+1):
   paths=[root/arm/'evaluation'/f'rng_before_stage{stage}_task{task}.pth' for arm in ('random','coreset')]
   for p in paths:hashes[str(p)]=sha(p)
   a,b=[torch.load(p,map_location='cpu') for p in paths]
   out.append({'seed':seed,'stage':stage,'task':task,'equal':{k:same(a[k],b[k]) for k in a}})
assert all(sha(Path(p))==h for p,h in hashes.items())
dest=Path(sys.argv[1]);assert not dest.exists()
dest.write_text(json.dumps({'rows':out,'hashes':hashes,'inputs_unchanged':True},indent=2)+'\n')
print(json.dumps(out,indent=2))
