"""Read-only final artifact acceptance; never trains or reruns rollouts."""
import json
import math
import hashlib
import subprocess
from collections import Counter
from pathlib import Path
import torch
import yaml

B = Path('/data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2')
R = B / 'run'
def read(p):
    return json.loads(p.read_text())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

c = read(R/'modal_contract.json')
def contract(d):
    assert d['modal_contract'] == c
    assert d['modal_fingerprint'] == c['fingerprint']

assert sha(R/'source.patch') == c['code_patch_sha256']
assert sha(B/'evidence/source.patch') == c['code_patch_sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(B/'source'),text=True).strip() == c['code_commit']
assert not subprocess.check_output(['git','diff','HEAD','--','libero/lifelong/models/complete_policy.py'],cwd=str(B/'source'))
original_status = subprocess.check_output(['git','status','--porcelain'],cwd='/data/mlr_ifa/src',text=True)
assert original_status == ''
configs = {}
for phase in ['export','adaptation','eval']:
    cfg = yaml.safe_load((R/('resolved_'+phase+'.yaml')).read_text())
    assert all(cfg['policy'][k] is True for k in ['use_vision','use_language','use_state'])
    assert cfg['seed'] == 42 and cfg['gamma'] == .7 and cfg['eval']['n_eval'] == 20
    assert cfg['train']['n_epochs'] == 100 and cfg['train']['batch_size'] == (1 if phase=='export' else 10)
    configs[phase] = {'sha256':sha(R/('resolved_'+phase+'.yaml')),'bool_modalities':True}

fm = read(R/'feature_manifest.json'); contract(fm)
assert fm['complete'] and fm['resolved_export_sha256'] == configs['export']['sha256']
windows = {6:997,7:889,8:904,9:1969}; selected={6:499,7:445,8:452,9:985}
for t,n in windows.items():
    entry=fm['tasks'][str(t)]
    assert entry['windows']==n and len(entry['files'])==n
    assert set(entry['demo_ids']) == {'demo_'+str(i) for i in range(10)}
    assert {f['source_index'] for f in entry['files']} == set(range(n))
    for f in entry['files']:
        assert sha(R/f['file']) == f['sha256']
        assert (R/f['file']).stat().st_size == f['bytes']

checkpoints=read(R/'checkpoints/manifest.json')
assert set(checkpoints)=={'6','7','8','9'}
official=Path('/data/mlr_ifa/checkpoints/goal_multitask_model_ep96.pth')
assert sha(official)==c['official_checkpoint_sha256']
op=torch.load(str(official),map_location='cpu')
osd={k.removeprefix('module.') if hasattr(str,'removeprefix') else (k[7:] if k.startswith('module.') else k):v for k,v in op['state_dict'].items()}
ref={k:v.shape for k,v in osd.items() if k.startswith(('temporal_transformer.','policy_head.'))}
assert len(ref)==100
del osd,op
parent=c['official_checkpoint_sha256']; global_step=0; summaries={}; buffer_counts={}
for t,n in windows.items():
    old=sum(selected[x] for x in range(6,t)); batches=math.ceil((n+old)/10)
    paths=sorted((R/'logs/epochs').glob('task%d_epoch*.json'%t)); assert len(paths)==100
    allsteps=[]
    for epoch,p in enumerate(paths):
        e=read(p); s=e['summary']; st=e['steps']
        assert s['epoch_index']==epoch and s['task']==t and s['optimizer_steps']==batches
        assert s['new_samples']==n and s['old_samples']==old and s['repeated_samples']==0
        assert s['per_task_counts']=={str(x):(n if x==t else selected[x]) for x in range(6,t+1)}
        assert len(st)==batches
        for row in st:
            global_step+=1
            assert row['global_step']==global_step
            assert all(math.isfinite(row[k]) for k in ['bc','ifa_raw','ifa_weighted','total','grad_norm'])
        assert s['global_step']==global_step
        allsteps.extend(st)
    total=100*batches
    for i,row in enumerate(allsteps):
        def lr(j):
            return 1e-4*(j/500 if j<500 else max(0,(total-j)/(total-500)))
        assert math.isclose(row['lr_used'],lr(i),abs_tol=1e-14)
        assert math.isclose(row['lr_next'],lr(i+1),abs_tol=1e-14)
    info=checkpoints[str(t)]; contract(info)
    p=Path(info['path']); assert sha(p)==info['sha256'] and info['parent_checkpoint_sha256']==parent
    cp=torch.load(str(p),map_location='cpu'); contract(cp)
    assert cp['task']==t and cp['epoch_index']==99 and cp['global_step']==global_step
    assert cp['parent_checkpoint_sha256']==parent and cp['feature_manifest_sha256']==sha(R/'feature_manifest.json')
    assert cp['scheduler']['last_epoch']==total and cp['optimizer']['param_groups'][0]['lr']==0
    assert cp['sampler']=={'epoch':99,'seed':0,'num_replicas':1,'rank':0}
    assert cp['rng'] and cp['optimizer']['state']
    sd={(k[7:] if k.startswith('module.') else k):v for k,v in cp['state_dict'].items()}
    assert set(sd)==set(ref)
    assert all(v.shape==ref[k] and torch.isfinite(v).all() for k,v in sd.items())
    load=read(R/('evaluation/load_stage%d.json'%t))
    assert load['coverage']==1 and load['loaded_keys']==100 and load['missing']==[] and load['unexpected']==[]
    assert load['spatial_keys_not_overwritten']==515
    parent=info['sha256']; del cp,sd
    bd=R/'buffer_manifests'/('after_task_%d'%t); bm=read(bd/'manifest.json'); contract(bm)
    assert bm['complete'] and bm['task']==t and bm['storage_rate']==.5
    counts=Counter(int(f['name'].split('__')[0][1:]) for f in bm['files'])
    assert dict(counts)=={x:selected[x] for x in range(6,t+1)}
    assert len({f['name'] for f in bm['files']})==len(bm['files'])
    for f in bm['files']: assert sha(bd/f['name'])==f['sha256']
    buffer_counts[str(t)]=dict(counts)
    summaries[str(t)]={'epochs':100,'optimizer_steps':total,'global_step':global_step,'checkpoint_sha256':info['sha256'],'load_coverage':1.0,'samples_per_epoch':n+old,'new_samples':n,'old_samples':old,'old_fraction':old/(n+old),'lr_final':0}
assert global_step==76100
assert len((R/'logs/train.jsonl').read_text().splitlines())==400
rows=[json.loads(s) for s in (R/'evaluation/rollouts.jsonl').read_text().splitlines()]
assert len(rows)==200 and len({(r['stage'],r['task_id'],r['init_state_id']) for r in rows})==200
matrix=[[None]*4 for _ in range(4)]; successes={}
for stage in range(6,10):
    for task in range(6,stage+1):
        cell=[r for r in rows if r['stage']==stage and r['task_id']==task]
        assert len(cell)==20 and {r['init_state_id'] for r in cell}==set(range(20))
        assert len({r['init_state_hash'] for r in cell})==20
        saved=read(R/('evaluation/stage%d_task%d.json'%(stage,task))); contract(saved)
        assert saved['episodes']==cell and saved['checkpoint_sha']==checkpoints[str(stage)]['sha256']
        semantic=read(R/('evaluation/online_modality_stage%d_task%d.json'%(stage,task)))
        assert semantic['status']=='PASS' and semantic['first3_tokens_exactly_unchanged']
        for r in cell:
            i=r['init_state_id']; assert r['kind']=='FORMAL' and r['training_seed']==42 and r['eval_seed']==42
            assert r['checkpoint_sha']==checkpoints[str(stage)]['sha256'] and r['modal_fingerprint']==c['fingerprint']
            assert r['env_batch_seed']==42+i//10 and r['worker_seed']==42+i//10+i%10
            assert r['exception'] is None and 1<=r['policy_steps']<=600
            assert r['success'] or (r['policy_steps']==600 and r['termination_reason']=='max_policy_steps')
        wins=sum(r['success'] for r in cell); matrix[stage-6][task-6]=wins/20
        successes['%d,%d'%(stage,task)]=wins
computed={'FWT':sum(matrix[i][i] for i in range(4))/4,
          'NBT':sum(sum(matrix[j][j]-matrix[i][j] for i in range(j+1,4))/(3-j) for j in range(3))/3,
          'AUC':sum(sum(matrix[i][j] for i in range(j,4))/(4-j) for j in range(4))/4,
          'FinalAvgSR':sum(matrix[3])/4}
recorded=read(R/'metrics.json')
assert all(math.isclose(v,recorded[k],abs_tol=1e-12) for k,v in computed.items())
gate=read(B/'smoke_v2_release3/test_gate.json'); contract(gate); assert gate['status']=='PASS'
assert read(R/'RUN_STATUS.json')['status']=='COMPLETED'
result={'status':'PASS','verification':'read-only artifacts; no new training or rollouts','training_epochs':400,'optimizer_steps':global_step,'formal_episodes':200,'valid_cells':10,'feature_windows':sum(windows.values()),'stages':summaries,'buffer_counts':buffer_counts,'success_counts_out_of_20':successes,'success_matrix':matrix,'metrics':computed,'configs':configs,'modal_fingerprint':c['fingerprint'],'source_patch_sha256':c['code_patch_sha256'],'official_sha256':c['official_checkpoint_sha256'],'original_repository_clean':True,'complete_policy_body_unchanged':True,'online_modality_cells_passed':10,'all_feature_buffer_checkpoint_hashes_verified':True}
(R/'final_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
