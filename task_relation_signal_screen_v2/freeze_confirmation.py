"""Seal identities, numerical rules and finite budget before new outcomes."""
import shutil
import time
import numpy as np
from .common import *
from .confirm_runtime import REPORT,DATA

def main():
    import h5py
    assert read(OUT/'INDEPENDENT_STAGE_A_AUDIT.json')['status']=='PASS'
    panel=read(OUT/'TASK_PANEL_MANIFEST.json');assert panel['status']=='FROZEN'
    assert read(OUT/'FINALISTS_FROZEN.json')['status']=='FROZEN'
    REPORT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    rules={}
    for key,e in read(OUT/'CONFIRMATION_DATA_MANIFEST.json')['tasks'].items():
        demos=e['demos']['D_signal'];assert len(demos)==10
        with h5py.File(e['path'],'r') as f:
            actions=np.concatenate([f['data'][d]['actions'][:] for d in e['demos']['A_train']])
        rules[key]=dict(D1=demos[:5],D2=demos[5:],action_std=np.maximum(actions.std(0),.05).tolist(),action_scale_source='A_train only; floor0.05 as development')
    freeze(REPORT/'DATA_RULES.json',dict(tasks=rules,data_manifest_sha256=sha(OUT/'CONFIRMATION_DATA_MANIFEST.json')))
    proto=read(OLD/'TRAINING_PROTOCOL.json')
    proto.update(dropout_stream='stable_seed(task_relation_signal_screen_v2,train,seed,target,call_index); shared by all source branches',source_calls=1000,target_calls=300,probe_effective=[1,5,20],max_probe_calls=50,formal_training_resumed=False)
    freeze(REPORT/'TRAINING_PROTOCOL.json',proto)
    freeze(REPORT/'EVALUATION_PROTOCOL.json',dict(source_models=18,ordered_seed_pairs=90,unique_model_task_seed_cells=234,trials_per_cell=40,total_trials=9360,source_calls=1000,target_calls=300,max_native_calls=50400,expected_native_calls=45000,paired_bootstrap=5000,maxT_simultaneous_confidence=.95,effect_band=.05,relative_MSE_improvement=.05,target_cells_minimum=4,rank_rule='same-target source ordering, plus same-source target damage ordering',decision_rule='highest predicted transfer >0.05 among sources with predicted damage <=0.05; otherwise base',native_evaluator='libero/lifelong/metric.py',native_action_sampling=True,initial_indices=list(range(40)),confirmation_seed_rule='fixed48/49/50 shared base, no independent pretraining claim',formal_training_resumed=False))
    paths=[OUT/'TASK_PANEL_MANIFEST.json',OUT/'CONFIRMATION_DATA_MANIFEST.json',OUT/'FINALISTS_FROZEN.json',OUT/'SIGNAL_REGISTRY.json',REPORT/'DATA_RULES.json',REPORT/'TRAINING_PROTOCOL.json',REPORT/'EVALUATION_PROTOCOL.json']
    freeze(REPORT/'FROZEN_INPUTS.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in paths},formal_training_resumed=False))
    # Correct bookkeeping without rewriting the initial resource snapshot.
    write(REPORT/'RESOURCE_CHECK.json',dict(free_bytes=shutil.disk_usage(DATA).free,checked_unix=time.time(),reserve_bytes=2*1024**3,expected_new_model_bytes=234*86572753,expected_feature_upper_bytes=6*3*5500*8*4*512*4,stage_A_calls_accounting='36 units +6 repeated units;2 branches each;50 max calls =4200 upper, actual21 per branch',stage_B_budget_changed=False,formal_training_resumed=False))
    assert shutil.disk_usage(DATA).free>35*1024**3,'CONFIRMATION_STORAGE_INSUFFICIENT'

if __name__=='__main__':main()
