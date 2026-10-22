"""CPU-only teacher scale/source audit. No forward, optimizer or environment."""
import json,hashlib,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
O=Path('/data/mlr_ifa/experiments/CAETv3_EFTR_OBJECT_seed41_20261022_collate01')
S=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/EFTR/source')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((O/'teachers/task6.json').read_text());cp=O/'teachers/task6.pth';assert sha(cp)==meta['sha256']
teacher=torch.load(cp,map_location='cpu');std=torch.cat([x['std'].reshape(-1) for x in teacher.values()]);assert len(teacher)==meta['count']
source=S/'libero/lifelong/models/policy_head.py';lines=source.read_text().splitlines();snip='\n'.join('%d: %s'%(i+1,x) for i,x in enumerate(lines) if 130<=i<=195)
cfg=json.loads((O/'resolved_ra_coreset.json').read_text())
policy=cfg.get('policy',cfg.get('cfg',{}).get('policy',{}))
selected_std=torch.stack([x['std'][x['mode']] for x in teacher.values()]); quantiles={str(j):[float(v) for v in torch.quantile(selected_std[:,j],torch.tensor([0.,.25,.5,.75,1.]))] for j in range(7)}
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='READONLY_SCALE_EVIDENCE',teacher_count=len(teacher),teacher_sha256=sha(cp),teacher_std_min=float(std.min()),teacher_std_max=float(std.max()),all_std_equal=bool((std==std[0]).all()),selected_std_quantiles_by_dimension=quantiles,source_class='TemporalGMMHead',source_path=str(source),source_sha256=sha(source),source_excerpt=snip,policy_config=policy,config_sha256=sha(O/'resolved_ra_coreset.json'),cuda_initialized=False,model_forwards=0,optimizer_updates=0,environment_steps=0,server_writes=0)))
