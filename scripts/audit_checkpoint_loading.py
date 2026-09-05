"""Read-only checkpoint compatibility and available task6 control inspection."""
import json
from pathlib import Path
import torch

root=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
def load(path):
    state=torch.load(path,map_location='cpu')['state_dict']
    return {k[7:] if k.startswith('module.') else k:v for k,v in state.items()}
pre=load(root/'checkpoints/object_multitask_model_ep99.pth')
required={k for k in pre if k.startswith(('temporal_transformer.','policy_head.'))}
assert required
rows=[]
for method in ('mlr','ifa'):
    for seed in (42,43,44):
        for stage in range(6,10):
            path=root/f'runs/final_paper_text_faithful/{method}/seed{seed}/run_001/mlrifa_{stage}_model_ep99.pth'
            raw=torch.load(path,map_location='cpu')['state_dict']
            assert all(k.startswith('module.') for k in raw), path
            state={k[7:]:v for k,v in raw.items()}
            missing=sorted(required-state.keys())
            unexpected=sorted(state.keys()-pre.keys())
            mismatched=[k for k,v in state.items() if k in pre and v.shape!=pre[k].shape]
            assert not missing and not unexpected and not mismatched,(path,missing,unexpected,mismatched)
            rows.append(dict(method=method,seed=seed,stage=stage,decoder_head_keys=len(required),checkpoint_keys=len(state),missing_trainable_keys=missing,unexpected_keys=unexpected,shape_mismatch=mismatched,omitted_pretrained_keys=len(pre.keys()-state.keys())))
            del state,raw
controls=[]
for seed in (42,43,44):
    new=root/f'runs/own_reference_control/ifa/seed{seed}/run_001/mlrifa_6_model_ep99.pth'
    if not new.exists():continue
    a=load(root/f'runs/final_paper_text_faithful/ifa/seed{seed}/run_001/mlrifa_6_model_ep99.pth')
    b=load(new)
    changed=[k for k in a if not torch.equal(a[k],b[k])]
    controls.append(dict(seed=seed,changed_tensors=len(changed),max_abs_difference=max((float((a[k]-b[k]).abs().max()) for k in changed),default=0)))
result=dict(checkpoint_loading=rows,task6_controls=controls,interpretation='CompletePolicy first strictly loads the full official pretrained model; partial lifelong checkpoints then replace temporal decoder and policy head. Missing frozen encoder keys are intentional, not randomly initialized. All 24 historical ep99 checkpoints have the expected module prefix and complete matching decoder/head keys.')
print(json.dumps(result,indent=2))
