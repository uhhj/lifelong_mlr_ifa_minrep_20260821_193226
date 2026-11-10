"""Metadata inventory only: no model construction, environment or server write."""
import hashlib,json,os,subprocess,time
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering02')
P=Path('/data/mlr_ifa/experiments/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering02')
ACTIVE=Path('/data/mlr_ifa/experiment_workspaces/S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1')
assert not Q.exists() and not P.exists(),'New reverse root must be unique'
J=lambda p:json.loads(Path(p).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
cache={}
def h(path):
    p=Path(path);s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    if key not in cache:cache[key]=sha(p)
    return cache[key]
assert h(ACTIVE/'CONTRACT.json')=='b54b0c2d8d8b490d6f7c394f7f99fa6689222861240454b4dd4eca870158b598'
results={}
for case,want in PARENTS.items():
    old=Path(want['main_mlr_parent']);align=Path(want['debr_align_parent'])
    bc=J(old/'resolved_random.json');ec=J(old/'resolved_eval.json') if (old/'resolved_eval.json').exists() else J(old/'random/evaluation/resolved_actual.json')
    dc=J(align/'resolved_ra_coreset.json');de=J(align/'resolved_eval.json')
    assert int(bc['seed'])==want['seed'] and int(dc['seed'])==want['seed']
    assert h(bc['pretrain'])==h(dc['pretrain'])==want['official_pretrain_sha256']
    assert bc['data']['task_order_index']==dc['data']['task_order_index']==0
    assert bc['train']['n_epochs']==dc['train']['n_epochs']==100 and bc['train']['batch_size']==dc['train']['batch_size']==10
    tasks={}
    for task in range(6,10):
        label=want['labels'][str(task)]
        assert h(label['path'])==label['sha256'] and h(label['binding_path'])==label['binding_sha256']
        assert h(label['hdf5'])==label['hdf5_sha256']
        bfiles=sorted((old/'features'/str(task)).glob('*.pt'))
        dfiles=sorted((align/'features'/str(task)).glob('*.pt'))
        assert len(bfiles)==len(dfiles)==label['total']
        bstore=old/'random/storage'/('task%d.json'%task)
        dstore=align/'ra_coreset/storage'/('task%d.json'%task)
        bbuf=Path(bc['buffer_dir'])
        dbuf=Path(dc['buffer_dir'])
        bmembers=sorted(p.name for p in bbuf.glob('t%d__*.pt'%task))
        dmembers=sorted(p.name for p in dbuf.glob('t%d__*.pt'%task))
        metadata_files=[]
        for base in [old/'random',align/'ra_coreset']:
            metadata_files.append([str(p.relative_to(base)) for p in base.rglob('*.json') if ('storage' in p.parts or 'select' in str(p.relative_to(base)).lower())][:40])
        tasks[str(task)]=dict(current_count=len(bfiles),baseline_feature_root=str(old/'features'/str(task)),debr_feature_root=str(align/'features'/str(task)),first_feature_sha_baseline=h(bfiles[0]),first_feature_sha_debr=h(dfiles[0]),all_feature_names_equal=[p.name for p in bfiles]==[p.name for p in dfiles],
            random_buffer=str(bbuf),random_buffer_members=bmembers,ra_buffer=str(dbuf),ra_buffer_members=dmembers,
            random_storage_path=str(bstore),random_storage=J(bstore) if bstore.exists() else None,
            ra_storage_path=str(dstore),ra_storage=J(dstore) if dstore.exists() else None,selection_metadata_inventory=metadata_files,
            labels_path=label['path'],labels_sha256=label['sha256'],labels_binding_path=label['binding_path'],labels_binding_sha256=label['binding_sha256'])
    results[case]=dict(main_mlr_parent=str(old),debr_align_parent=str(align),debr_align_source=want['debr_align_source'],main_mlr_config=bc,main_mlr_eval_config=ec,debr_config=dc,debr_eval_config=de,official_pretrain_sha256=want['official_pretrain_sha256'],tasks=tasks)
capacity={}
for mount in ['/','/data']:
    v=os.statvfs(mount);capacity[mount]=dict(available=v.f_bavail*v.f_frsize,free=v.f_bfree*v.f_frsize)
assert capacity['/']['available']>=1024**3+64*1024**2
active_status=J(ACTIVE/'STATUS.json')
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True)
print(json.dumps(dict(status='READONLY_REVERSE_INPUT_INVENTORY_NOT_PREPARATION_ACCEPTANCE',observed=time.time(),parents=results,capacity=capacity,gpu=gpu.stdout,current_queue_status=active_status,new_optimizer=0,new_environments=0,new_model_forward=0,server_writes=0)))
