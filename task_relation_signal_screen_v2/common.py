import csv
import os
from pathlib import Path
from task_relation_forecast_v1.common import read, write, freeze, sha, stable_seed

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'reports/task_relation_forecast_v1'
OUT=ROOT/'reports/task_relation_signal_screen_v2'
STORE=ROOT/'diagnostic_storage/task_relation_signal_screen_v2'
TASKS=(6,7,8,9)
SEEDS=(45,46,47)
PAIRS=[(k,s,t) for k in SEEDS for s in TASKS for t in TASKS if s!=t]
SMOKE=[(k,s,t) for k in SEEDS for s,t in ((6,7),(8,9))]
PACKAGES={
 'S01':['visual','language'], 'S02':['mmd'], 'S03':['gn_overlap'],
 'S04':['affinity'], 'S05':['action_improvement','gripper_improvement'],
 'S06':['affinity_lower_tail','affinity_worst_third','affinity_gripper_switch'],
 'S07':['linear_probe'], 'S08':['label_mse'], 'S09':['logme_difference'],
 'S10':['coverage','conditional_action','conditional_gripper'], 'S11':['gradient_cosine'],
 'S12':['risk_mean','risk_tail','risk_positive'],
 'S13':['relative_progress5','source_damage5','source_tail5'],
 'S14':['relative_progress20','source_damage20','source_tail20','damage_5_to_20']}

def table(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)

def npz(path,**arrays):
    import numpy as np
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.npz');np.savez_compressed(tmp,**arrays);os.replace(tmp,path)

def unit(k,s,t):return 's%d_%d_%d'%(k,s,t)
