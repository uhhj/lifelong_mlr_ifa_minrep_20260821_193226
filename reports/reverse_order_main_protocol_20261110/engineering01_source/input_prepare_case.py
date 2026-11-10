"""CPU-only private input aliases and order-specific IFA; no model or env."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',MUJOCO_GL='egl')
from pathlib import Path
import copy,hashlib,json,math,random,sys,time
Q=Path(__file__).resolve().parent
case=sys.argv[1];R=Q/'cases'/case
J=lambda p:json.loads(Path(p).read_text())
inp=J(R/'INPUT.json');O=Path(inp['input_root']);parent=inp['parent'];method=inp['method']
assert not (R/'PREPARED_CASE.json').exists(),'Do not repeat completed input preparation'
O.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(Q));sys.path.insert(0,str(R))
sys.path.insert(0,str(R/'source') if method=='MLR' else inp['source_root'])
os.environ.update(CORESET_OUT=str(O),CORESET_SEED=str(inp['seed']),CAET_VARIANT='ALIGN')
import numpy as np
import torch
torch.set_num_threads(1)
from order import install,verify_mapping,ORIGINAL_TASKS
install()
from easydict import EasyDict
from coreset_exp.common import write,sha,jsonable
from libero.lifelong.algos.mlr import build_ifa_similarity_matrix,_sample
tic=time.time();external={};current={};labels={};selections={};metadata={}
assert not torch.cuda.is_initialized()
def bind(path,expected=None):
    path=Path(path);s=path.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    if key not in hashes:hashes[key]=sha(path)
    value=hashes[key]
    assert expected is None or value==expected,(str(path),'input changed')
    external[str(path)]=value
    return value
hashes={}
baseline_features=J(Path(parent['main_mlr_parent'])/'FEATURES.json')
bind(Path(parent['main_mlr_parent'])/'FEATURES.json')
def alias(dst,src):
    dst=Path(dst);src=Path(src);dst.parent.mkdir(parents=True,exist_ok=True)
    assert not dst.exists() and not dst.is_symlink(),dst
    os.symlink(str(src),str(dst));assert dst.resolve()==src.resolve()
    return dst
for position in range(6,10):
    original=ORIGINAL_TASKS[position];row=parent['tasks'][str(original)]
    base=Path(row['baseline_feature_root'] if method=='MLR' else row['debr_feature_root'])
    paths=sorted(base.glob('*.pt'));assert len(paths)==inp['current_counts'][str(position)]
    selected=row['random_selected' if method=='MLR' else 'ra_selected']
    original_store=Path(row['random_storage_path' if method=='MLR' else 'ra_storage_path'])
    source_store=J(original_store);bind(original_store)
    assert source_store['selected']==selected
    if method=='MLR':assert [Path(x).name for x in _sample([str(p) for p in paths],len(selected),random.Random(42))]==[Path(x['source']).name for x in selected]
    current[position]=[]
    selected_by_name={Path(x['source']).name:x['sha256'] for x in selected}
    if method=='DEBR':
        bind(row['labels_path'],row['labels_sha256']);bind(row['labels_binding_path'],row['labels_binding_sha256'])
        original_labels=torch.load(row['labels_path'],map_location='cpu')
        assert set(original_labels)==set(p.name for p in paths)
        labels[position]={}
    for path in paths:
        expected=baseline_features['hashes'][str(original)+'/'+path.name] if method=='MLR' else original_labels[path.name]['sample_sha256']
        digest=bind(path,expected)
        assert path.name not in selected_by_name or digest==selected_by_name[path.name]
        logical=alias(O/'features'/str(position)/path.name,path)
        current[position].append(dict(path=str(logical),sha256=digest,sample=path.name,task=position,original_task=original))
        if method=='DEBR':
            value=copy.deepcopy(original_labels[path.name]);assert value['sample_sha256']==digest and value['task']==original
            value['task']=position;value['original_task']=original
            labels[position][path.name]=value
    selections[position]=[dict(sample=Path(x['source']).name,sha256=x['sha256'],original_task=original,task=position) for x in selected]
    assert len(selections[position])==inp['memory_counts'][str(position)]
    metadata[str(position)]=dict(original_task=original,current_count=len(paths),memory_count=len(selected),old_storage_sha256=sha(original_store),selected=selections[position])
    if method=='DEBR':
        target=O/'effects/current'/('task%d'%position);target.mkdir(parents=True)
        torch.save(labels[position],target/'labels.pth')
        write(target/'AUDIT.json',dict(labels_sha256=sha(target/'labels.pth'),current_count=len(paths),original_task=original,renumbering_only=True,source_labels_sha256=row['labels_sha256']))
        external[str(target/'labels.pth')]=sha(target/'labels.pth')
        memory={}
        for x in selections[position]:
            src=Path(row['debr_feature_root'])/x['sample']
            alias(O/'ra_coreset/buffer'/('t%d__%s'%(position,x['sample'])),src)
            memory['t%d__%s'%(position,x['sample'])]=labels[position][x['sample']]
        target=O/'effects/memory'/('task%d.pth'%position);target.parent.mkdir(parents=True,exist_ok=True)
        torch.save(memory,target);write(target.with_suffix('.json'),dict(sha256=sha(target),source_original_task=original))
        external[str(target)]=sha(target)

cfg=copy.deepcopy(inp['train_config']);ec=copy.deepcopy(inp['eval_config'])
mapping=verify_mapping(EasyDict(cfg));assert cfg['data']['task_order_index']==ec['data']['task_order_index']==21
assert all(mapping[str(i)]['original_task']==i for i in range(6))
bind(cfg['pretrain'],parent['official_pretrain_sha256'])
write(R/'TASK_MAPPING.json',dict(status='PASS_ACTUAL_BASE_AND_INCREMENTAL_MAPPING',mapping=mapping,original_order=list(range(10)),new_order=ORIGINAL_TASKS))
stage_inputs={};population={};valid={};old_valid={};manifests={}
for stage in range(6,10):
    # New task-independent selected members, remapped to chronological labels;
    # never reuse a forward-order stage mixture or IFA pair selection.
    old=[]
    stage_buffer=O/'stage_preparation'/('stage%d'%stage)/'buffer';stage_buffer.mkdir(parents=True)
    for prior in range(6,stage):
        original=ORIGINAL_TASKS[prior];row=parent['tasks'][str(original)]
        root=Path(row['baseline_feature_root'] if method=='MLR' else row['debr_feature_root'])
        for selection in selections[prior]:
            path=alias(stage_buffer/('t%d__%s'%(prior,selection['sample'])),root/selection['sample'])
            logical=(O/'ra_coreset/buffer'/path.name) if method=='DEBR' else path
            old.append(dict(path=str(logical),sha256=selection['sha256'],sample=selection['sample'],task=prior,original_task=original))
    old.sort(key=lambda x:Path(x['path']).name)
    rows=old+copy.deepcopy(current[stage]);random.Random(inp['seed']).shuffle(rows)
    assert len(rows)==inp['population'][str(stage)]
    if method=='DEBR':
        for index,value in enumerate(rows):
            label=labels[value['task']][value['sample']]
            value.update(index=index,demo=label['demo'],u=label['u'],bc_valid=label['bc_valid'])
        target=O/'ra_coreset/dataset_manifests'/('task%d.json'%stage)
        man=dict(task=stage,original_task=ORIGINAL_TASKS[stage],population=len(rows),rows=rows)
        write(target,man);manifests[str(stage)]=dict(path=str(target),sha256=sha(target))
        valid[stage]=sum(x['bc_valid'] for x in rows)
        old_valid[stage]=sum(x['bc_valid'] for x in rows if x['task']<stage)
    else:
        # The inherited training loader builds this same logical sequence once
        # preceding tasks have actually populated its own fresh Random buffer.
        valid[stage]=len(rows);old_valid[stage]=len(old)
        target=R/'manifests'/('task%d.json'%stage)
        write(target,dict(task=stage,original_task=ORIGINAL_TASKS[stage],population=len(rows),rows=rows))
        manifests[str(stage)]=dict(path=str(target),sha256=sha(target))
    info=build_ifa_similarity_matrix(stage,str(O/'features'),str(stage_buffer))
    assert info['task_ids']==list(range(6,stage+1))
    public=dict(info=jsonable(info),source='RECOMPUTED_FROM_REVERSE_CURRENT_AND_OWN_SELECTED_OLD_FEATURES',stage=stage,original_task=ORIGINAL_TASKS[stage])
    write(R/('EXPECTED_IFA_%d.json'%stage),public)
    if method=='DEBR':write(O/'ra_coreset'/('ifa_task%d.json'%stage),public)
    stage_inputs[str(stage)]=dict(population=len(rows),batches=(len(rows)+9)//10,manifest_sha256=sha(target),ifa_sha256=sha(R/('EXPECTED_IFA_%d.json'%stage)),valid=valid[stage],old_valid=old_valid[stage])
    population[stage]=len(rows)

write(O/'resolved_eval.json',ec)
write(O/('resolved_random.json' if method=='MLR' else 'resolved_ra_coreset.json'),cfg)
write(R/'INPUT_COUNTS.json',dict(population=population,valid=valid,old_valid=old_valid))
write(R/'SELECTION_BINDING.json',dict(status='PASS_ORIGINAL_TASK_MEMBERS_PRESERVED',tasks=metadata,selection_is_order_independent_with_frozen_frontend=True))
if method=='DEBR':
    write(O/'ra_coreset/INITIAL_STATE.json',inp['initial_state'])
    # These actual files are selected by the same physical task mapping used in benchmark/data_setup.
    from libero.libero.benchmark import get_benchmark
    b=get_benchmark(cfg['benchmark_name'])(21);splits={}
    for stage in range(6,10):
        task=b.get_task(stage);path=Path(cfg['init_states_folder'])/task.problem_folder/task.init_states_file
        states=torch.load(path,map_location='cpu');bind(path)
        def ah(x):return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()
        splits[str(stage)]=dict(file_sha256=sha(path),training={str(i):ah(states[i]) for i in range(20,30)},formal={str(i):ah(states[i]) for i in range(20)},original_task=ORIGINAL_TASKS[stage])
    write(R/'INITIAL_SPLIT.json',splits)
    # The runtime only needs this parent's immutable source identities; old science stays read-only.
    parent_contract=J(Path(parent['debr_align_parent'])/'CONTRACT.json')
    write(O/'CONTRACT.json',dict(source_sha256=parent_contract['source_sha256'],reverse_order=ORIGINAL_TASKS,scope='NEW_INPUT_NAMESPACE_PARENT_SOURCE_IDENTITIES'))
else:
    # cfg/evaluation contract semantics are filled after all source/metadata files
    # are in place; no train function is called by this preparation process.
    pass
inp['manifests']=manifests;write(R/'INPUT.json',inp)
write(R/'EXTERNAL_FILES.json',external)
assert not torch.cuda.is_initialized()
result=dict(status='PASS_CPU_INPUT_PREPARATION_NOT_MODEL_OR_GPU_ADMISSION',case=case,method=method,stages=stage_inputs,order=ORIGINAL_TASKS,accepted_budget=sum(v['batches']*100 for v in stage_inputs.values()),external_files=len(external),seconds=time.time()-tic,new_optimizer=0,new_environments=0,new_model_forward=0,cuda_initialized=False)
assert result['accepted_budget']==inp['accepted_budget']
write(R/'PREPARED_CASE.json',result);print(json.dumps(result))
