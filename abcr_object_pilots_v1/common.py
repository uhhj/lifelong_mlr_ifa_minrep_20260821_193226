import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(path,data,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)
    if path.exists() and immutable:
        assert json.loads(path.read_text())==data,'FROZEN_BINDING_MISMATCH: '+str(path)
        return
    tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'w',encoding='utf-8') as f:
        f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def table(path,rows):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def npz(path,**arrays):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.npz')
    with open(tmp,'wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def normalize(a):
    a=np.asarray(a,dtype=np.float64)
    norms=np.linalg.norm(a,axis=-1,keepdims=True)
    assert np.isfinite(a).all() and (norms>0).all()
    return a/norms

PROTOCOL=dict(experiment='abcr_object_pilots_v1',suite='LIBERO_OBJECT',base_tasks=list(range(6)),
    incremental_tasks=[6,7,8,9],checkpoint='object_multitask_model_ep99.pth',
    trajectory_selection='numeric demo0..4 for each old task; demo0..9 for current incremental task; every observed timestep',
    diagnostic_pool='Includes base-task histories per this pilot design; NOT native MLR replay (which starts empty at task6). No formal buffer mutation.',
    state='9D measured joint_states(7) + gripper_states(2); base-pool mean/std, std floor1e-6; Euclidean distance',
    visual='beta*cos(eih)+(1-beta)*cos(agent), frozen pre-FiLM image_proj features',
    beta=[.3,.5,.7],primary_beta=.5,state_quantile=.15,visual_quantile=.85,
    quantile_method='linear, all cross-task pairs of this stage, float64, exact not subsampled',
    action='RMS of normalized differences over q*7 commands; motion scales base IQR floor0.1, gripper scale2 (full command range); gripper conflict also reported separately',
    action_q=4,action_quantiles=[.25,.75],
    action_time='obs[t] recorded AFTER action[t]; genuine future starts action[t+1]; no terminal padding',
    p1_gate=dict(state_anchor_coverage=.30,share_anchor_coverage_primary=.10,bif_anchor_coverage_primary=.10,
                 lower_design_band=.08,denominator='all current-task observed timesteps, including unavailable future tails'),
    p1_interpretation='Relative quantile coverage is descriptive; cannot certify comparable physical states or true share/conflict. Per-beta results, no best-beta selection.',
    p2_methods=['H_full_cosine','C_physical_gate_then_visual','unmod_visual_only','H_same_physical_gate'],
    p2_C='Ds<=tau_s candidates then max visual score; no candidate => unavailable; tie smallest global old index',
    p2_H='cosine of flattened native CompletePolicy 4x512 output; actual config use_state=False makes fourth token state-FiLM(eih visual), not a proprioception token',
    p2_confounds='C uses proprioception: C-vs-H EEF advantage does not establish FiLM causality. Visual-only and equally physically gated H controls are mandatory.',
    p3_horizons=[1,2,3,4,5,6],p3_thresholds='same stage/beta q4 tau_plus/tau_minus for every horizon, no per-q quantile retuning',
    p3_label='short-horizon action divergence surrogate, not independently verified false sharing',
    p3_q_selection='Report all q; do not declare global optimal q or tune on best beta',
    statistics='current trajectory as paired unit; old endpoints reused; CIs descriptive not unbiased task-population inference',
    random_seed=42,bootstrap_repeats=2000,plot_sample_pairs=2000,
    p4='DEFERRED_BY_USER; no sampling or training P4 now',training_active=False,formal_training_resumed=False)
