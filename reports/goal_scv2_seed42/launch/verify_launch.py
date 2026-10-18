"""Local manifest and frozen-source audit; no training or environments."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;E=R/'launch_evidence'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p):return json.loads(p.read_text(encoding='utf-8'))
manifest=js(E/'PACKAGE_MANIFEST.json')
for n,r in manifest['files'].items():assert (E/n).stat().st_size==r['bytes'] and sha(E/n)==r['sha256'],n
c=js(E/'CONTRACT.json');gpu=js(E/'GPU_PREFLIGHT.json');launch=js(E/'LAUNCH.json')
assert c['version']=='SCv2_GOAL_seed42_v1' and c['seed']==42
assert c['epochs']==400 and c['optimizer_steps']==76100 and c['formal_records']==200
assert gpu['status']=='PASS' and gpu['isolated_optimizer_calls']==4 and gpu['environment_steps']==0
assert gpu['contract_sha256']==sha(E/'CONTRACT.json')==launch['contract_sha256']
assert launch['preflight_sha256']==sha(E/'GPU_PREFLIGHT.json')
for n,h in c['source_sha256'].items():assert sha(R/'source'/n)==h,n
for name in ['resolved_sc_coreset.json','resolved_eval.json']:
    cfg=js(E/name);assert cfg['seed']==42 and cfg['gamma']==.7
    assert all(cfg['policy'][k] is True for k in ['use_state','use_language','use_vision'])
    assert cfg['train']['batch_size']==10 and cfg['eval']['n_eval']==20
for t in range(6,10):
    off=js(E/'offline_selection'/f'task{t}.json');sel=js(E/'selection_preflight'/f'task{t}.json')
    assert off['original_ids']==sel['original_ids'] and off['selected_ids']==sel['selected_ids']
    assert gpu['selection'][str(t)]['offline_exact']
result=dict(status='LAUNCH_ACCEPTANCE_PASS_NOT_EXPERIMENT_COMPLETED',seed=42,formal_budget=dict(epochs=400,steps=76100,checkpoints=4,records=200),
    gpu_preflight=gpu,contract_sha256=sha(E/'CONTRACT.json'),package_files_verified=len(manifest['files']),
    source_files_verified=len(c['source_sha256']),source_code_changes=js(R/'CHANGED_SOURCE.json'),
    evaluation='original per-stage global RNG, reuse existing baseline outcomes',automatic_extension_gate=dict(AUC='>63/80',OldFinal='>=41/60'))
(R/'LOCAL_LAUNCH_ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
