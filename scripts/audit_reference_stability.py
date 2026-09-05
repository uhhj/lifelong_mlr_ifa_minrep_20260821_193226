"""Read-only full cache check of task language reference consistency."""
import json
from pathlib import Path
import torch

root=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
rows=[]
references={}
for seed in (42,43,44):
    for task in range(6,10):
        cache=root/f'features/object_seed{seed}/{task}'
        files=sorted(cache.glob('*.pt'))
        reference=torch.load(cache/'0.pt',map_location='cpu')['feature'][0,0,0,:].float()
        references[seed,task]=reference
        differing=0
        max_difference=0.0
        for path in files:
            feature=torch.load(path,map_location='cpu')['feature']
            # CompletePolicy.spatial_encode saves [1, time, modality, embedding].
            language=feature[0,:,0,:].float()
            assert torch.isfinite(language).all(),path
            difference=float((language-reference).abs().max())
            differing+=int(difference!=0)
            max_difference=max(max_difference,difference)
        row=dict(seed=seed,task=task,files=len(files),feature_shape=list(feature.shape),different_from_task_reference=differing,max_abs_difference=max_difference)
        rows.append(row)
cross_seed={str(task):{str(seed):float((references[seed,task]-references[42,task]).abs().max()) for seed in (43,44)} for task in range(6,10)}
print(json.dumps(dict(scope='All cached files and all stored time positions; only language modality compared; no cache writes',within_task=rows,cross_seed_reference_max_abs_difference=cross_seed),indent=2))
