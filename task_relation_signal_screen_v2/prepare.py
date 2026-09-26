"""Read-only historical provenance and label-blind measurement registry."""
import json
import platform
import shutil
import subprocess
import time
import numpy as np
from .common import *

def main():
    import torch
    OUT.mkdir(parents=True,exist_ok=True);STORE.mkdir(parents=True,exist_ok=True)
    history='c04dd0b30ffe30a21396ce27ee18639904eb3de8'
    named=['DATA_MANIFEST.json','TRAINING_PROTOCOL.json','SIGNALS.csv','TARGETS.csv','ROLLOUTS.csv','BASE_EXPOSURE_AUDIT.json']
    verified={}
    for name in named:
        saved=subprocess.check_output(['git','show',history+':reports/task_relation_forecast_v1/'+name],cwd=str(ROOT))
        import hashlib
        assert hashlib.sha256(saved).hexdigest()==sha(OLD/name),name
        verified[name]=sha(OLD/name)
    deps=read(OLD/'RUNTIME_PROVENANCE.json');provenance={}
    for rel,meta in deps.items():
        actual=sha(ROOT/rel);assert actual==meta['sha256'],('Runtime drift',rel)
        provenance[rel]=dict(path=str(ROOT/rel),sha256=actual,git_tracked=meta['git_tracked'])
    assets=[]
    from task_relation_forecast_v1 import runtime as rt
    for k in SEEDS:
        for s in TASKS:
            p=rt.model_path(k,'source',source=s,step=1000)
            expected=read(OLD/'signals'/unit(k,s,next(t for t in TASKS if t!=s))/'SIGNALS.json')['source_sha256']
            assert p.exists() and sha(p)==expected
            assets.append(dict(seed=k,task=s,path=str(p),sha256=expected,bytes=p.stat().st_size))
    write(OUT/'SOURCE_PROVENANCE.json',dict(historical_commit=history,actual_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),historical_files=verified,runtime=provenance,assets=assets,formal_training_resumed=False))
    write(OUT/'EXPERIMENT_CONTEXT.json',dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),disk_free=shutil.disk_usage(ROOT).free,formal_training_resumed=False,diagnostic_training_performed=False))
    manifest=read(OLD/'DATA_MANIFEST.json');splits={};windows={};scales={}
    import h5py
    for task in TASKS:
        e=manifest['tasks'][str(task)];d=e['demos'];assert not(set(d['A_train'])&set(d['D_signal']) or set(d['D_signal'])&set(d['E_eval']))
        ds=sorted(d['D_signal'],key=lambda demo:stable_seed('screen_v2',task,demo))
        splits[str(task)]=dict(D1=ds[:len(ds)//2],D2=ds[len(ds)//2:],original_demos=d)
        rows=e['windows']['D_signal'];gradient=[]
        for demo in ds:
            idx=[i for i,r in enumerate(rows) if r['trajectory_id']==demo]
            quota=32//len(ds)+(ds.index(demo)<32%len(ds))
            gradient+=list(np.array(idx)[np.linspace(0,len(idx)-1,quota,dtype=int)])
        windows[str(task)]=dict(signal=rows,gradient_indices=[int(i) for i in gradient])
        with h5py.File(e['path'],'r') as f:
            actions=np.concatenate([f['data/'+demo+'/actions'][:] for demo in d['A_train']])
        scales[str(task)]=dict(mean=actions.mean(0).tolist(),std=np.maximum(actions.std(0),.05).tolist(),source='existing A_train only; per action dimension floor0.05',file=e['path'])
    freeze(OUT/'DEMO_SPLITS.json',splits);freeze(OUT/'WINDOW_MANIFEST.json',windows);freeze(OUT/'ACTION_SCALES.json',scales)
    protocol=dict(status='MEASUREMENT_DEFINITIONS_FROZEN_NOISE_CALIBRATION_PENDING',protocol_sha256=sha(OUT/'PROTOCOL.md'),stage_A_pairs=PAIRS,technical_units=SMOKE,packages=PACKAGES,stage_A_new_rollouts=0,probe_q=[1,5,20],max_probe_calls=50,gn_rank=16,prediction_alpha=1.,alpha_sensitivity=[.1,10.],z_clip=5,prediction_clip=[-1,1],effect_band=.05,stage_B=dict(suites={'libero_goal':3,'libero_spatial':3},seeds=[48,49,50],source_steps=1000,target_steps=300,rollouts_per_cell=40,total_rollouts=9360,stage_B_started=False),formal_training_resumed=False)
    freeze(OUT/'PROTOCOL.json',protocol)
    freeze(OUT/'MEASUREMENT_RULES.json',dict(tau_MSE=1e-8,feature_std_floor=1e-8,action_std_floor=.05,readout_alpha=1.,readout_objective='mean over samples and action dimensions + ||W||_F^2; intercept unpenalized',mmd='biased RBF MMD, train-half pooled median squared-distance bandwidth, average two held halves',S10='base image_encoder+image_proj before FiLM (no language), masked frame mean + terminal proprioception; D_signal only',S12_deadband='32*float32_eps*max(1,abs(window NLL)), historical rule; no effect-tuned adjustment',missing='raw null+availability; folds use training mean (0 standardized), no extra predictive masks; whole unavailable packages never ranked as available',noise_rule='scale_floor=max(1e-8,10*max_abs_technical_repeat_difference); constant_floor=max(1e-8,10*max_abs_technical_repeat_difference); only six label-blind units; freeze before analyze-dev',source_labels_read_by_measurement=False,representation='S02/S07/S09 decoder output valid-frame mean; official GMM NLL still terminal output',readout_combination='only A+S07 extra descriptive; A+C20 extra descriptive; C20 optionally selected best C0 with <=6 extra dimensions',development_resampling=1000,group_permutations=1000))
    write(OUT/'RESOURCE_BUDGET.json',dict(stage_A_max_native_calls=36*50+12*50,stage_A_new_rollouts=0,stage_B_native_calls=45000,stage_B_upper_calls=50400,stage_B_rollouts=9360,stage_B_dataset_status='GOAL/SPATIAL data absent from current data directory; audit/download still required',stage_B_budget_frozen=True,disk_free_bytes=shutil.disk_usage(ROOT).free,eta_basis='v1 rollout throughput extrapolation ~57h; not a promise; new-suite smoke required',formal_training_resumed=False))
    write(OUT/'STATUS.json',dict(stage='PREPARED',static_complete=0,probe_complete=0,registry_noise_frozen=False,formal_training_resumed=False))

if __name__=='__main__':main()
