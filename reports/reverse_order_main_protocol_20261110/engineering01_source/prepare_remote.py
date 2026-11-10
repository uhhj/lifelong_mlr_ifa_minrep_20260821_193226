"""Finish staged CPU inputs and freeze once; does not launch the controller."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
Q=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1')
sys.path.insert(0,str(Q))
from reverse_common import read,sha,write,environment,PY
assert (Q/'STAGED.json').exists() and not (Q/'CONTRACT.json').exists() and not (Q/'LAUNCH.json').exists()
spec=read(Q/'SPEC.json');prepared={}
for case in spec['cases']:
    r=Q/'cases'/case
    assert not (r/'PREPARED_CASE.json').exists(),'Do not blindly repeat a prepared case'
    env=environment();env['CUDA_VISIBLE_DEVICES']=''
    p=subprocess.run([PY,'-B',str(Q/'input_prepare_case.py'),case],cwd=Q,env=env,capture_output=True,text=True)
    write(r/'PREPARATION_ATTEMPT.json',dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
    assert p.returncode==0,('CPU input preparation failed; preserve outputs before repair',case,p.stderr[-3000:])
    result=json.loads(p.stdout.strip().splitlines()[-1]);assert result['status']=='PASS_CPU_INPUT_PREPARATION_NOT_MODEL_OR_GPU_ADMISSION'
    prepared[case]=result
    # Actual Python 3.8 imports of complete method paths, never their main().
    inp=read(r/'INPUT.json')
    modules=['coreset_exp.train','coreset_exp.evaluate'] if inp['method']=='MLR' else ['runtime','updates','train','evaluate','collect_native','gate_admission','support_memory']
    code="import os,sys,json,importlib;from pathlib import Path;Q=Path(%r);R=Path(%r);sys.path[:0]=[str(R),%r,str(Q)];os.environ.update(REVERSE_CASE_ROOT=str(R),CORESET_OUT=%r,CORESET_SEED=%r,CAET_VARIANT='ALIGN');from order import install;install();import torch;torch.set_num_threads(1);[importlib.import_module(n) for n in %r];assert not torch.cuda.is_initialized();print(json.dumps(dict(status='PASS_ACTUAL_CPU_IMPORT_NO_SCIENCE',modules=%r,new_optimizer=0,new_environment=0,new_model_forward=0)))"%(str(Q),str(r),str(r/'source') if inp['method']=='MLR' else inp['source_root'],inp['input_root'],str(inp['seed']),modules,modules)
    check=subprocess.run([PY,'-B','-c',code],cwd=r,env=env,capture_output=True,text=True)
    write(r/'IMPORT_ATTEMPT.json',dict(exit=check.returncode,stdout=check.stdout,stderr=check.stderr))
    assert check.returncode==0,('Actual import failed; no training started',case,check.stderr[-3000:])
    write(r/'IMPORT_ADMISSION.json',json.loads(check.stdout.strip().splitlines()[-1]))

children={};frozen={};public={}
for case in spec['cases']:
    r=Q/'cases'/case;inp=read(r/'INPUT.json');o=Path(inp['input_root'])
    registry=read(r/'EXTERNAL_FILES.json')
    if inp['method']=='MLR':
        source=r/'source';sources={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.yml'] and '__pycache__' not in p.parts}
        modal=read(Path(inp['parent']['main_mlr_parent'])/'MODAL_CONTRACT.json')
        modal.update(source_sha256=sources,reverse_order=inp['order'],reverse_order_index=21,scope='NEW_REVERSE_MAIN_MLR_SOURCE_AND_IDENTICAL_OFFICIAL_FRONTEND')
        assert modal['flags']==dict(vision=True,language=True,state=True)
        write(o/'MODAL_CONTRACT.json',modal)
        write(o/'OLD_INPUTS_READONLY.json',dict(files=registry,source_main_parent=inp['parent']['main_mlr_parent']))
        cfgs={str(o/n):sha(o/n) for n in ['resolved_random.json','resolved_eval.json']}
        native_contract=dict(source_sha256=sources,modal_sha256=sha(o/'MODAL_CONTRACT.json'),old_readonly_sha256=sha(o/'OLD_INPUTS_READONLY.json'),inputs={inp['train_config']['pretrain']:inp['parent']['official_pretrain_sha256']},configs=cfgs,optimizer_steps=inp['accepted_budget'],order=inp['order'])
        write(o/'CONTRACT.json',native_contract)
        old_features=read(Path(inp['parent']['main_mlr_parent'])/'FEATURES.json')
        feature_sha={str(p.relative_to(o/'features')):sha(p) for p in (o/'features').rglob('*.pt')}
        features=dict(counts={str(k):v for k,v in inp['current_counts'].items()},demos={str(i):old_features['demos'][str(inp['order'][i])] for i in range(6,10)},hashes=feature_sha,modal_sha256=sha(o/'MODAL_CONTRACT.json'),contract_sha256=sha(o/'CONTRACT.json'),order=inp['order'])
        write(o/'FEATURES.json',features)
    else:
        # Freshly verify dependencies of the exact accepted S build/evaluator.
        source=Path(inp['source_root']);pc=read(Path(inp['parent']['debr_align_parent'])/'CONTRACT.json')
        for relative,h in pc['source_sha256'].items():
            assert sha(source/relative)==h,relative
            registry[str(source/relative)]=h
        write(r/'EXTERNAL_FILES.json',registry)
    private_metadata={str(p):sha(p) for p in o.rglob('*.json') if p.is_file()}
    case_files={str(p):sha(p) for p in r.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.yml','.json'] and p.name!='CONTRACT.json' and '__pycache__' not in p.parts}
    case_files.update(private_metadata)
    write(r/'CONTRACT.json',dict(case=case,order=inp['order'],files=case_files,expected_accepted=inp['accepted_budget'],expected_physical=inp['physical_budget'],input_external_registry_sha256=sha(r/'EXTERNAL_FILES.json')))
    children[case]=dict(contract_file=str((r/'CONTRACT.json').relative_to(Q)),contract_sha256=sha(r/'CONTRACT.json'))
    # These are source/metadata/hash text only. Private label/feature/RNG tensors never leave the server.
    for p in r.rglob('*.json'):
        if p.is_file():public[str(p.relative_to(Q))]=p.read_text()
    for p in r.rglob('*.yaml'):
        if p.is_file():public[str(p.relative_to(Q))]=p.read_text()
for p in Q.rglob('*'):
    if p.is_file() and p.suffix in ['.py','.yaml','.yml','.json'] and p.name not in ['CONTRACT.json','STAGED.json'] and '__pycache__' not in p.parts:
        frozen[str(p.relative_to(Q))]=sha(p)
contract=dict(name=spec['name'],order=spec['order'],files=frozen,cases=children,storage_peak_bytes=14*1024**3,epochs=4800,accepted=1156800,production=1636500,admission=666,training_collection=240,formal=2400)
write(Q/'CONTRACT.json',contract);public['CONTRACT.json']=(Q/'CONTRACT.json').read_text()
assert len(prepared)==12 and sum(x['accepted_budget'] for x in prepared.values())==1156800
write(Q/'PREPARATION_COMPLETE.json',dict(status='PASS_PREPARED_FROZEN_NOT_LAUNCHED',contract_sha256=sha(Q/'CONTRACT.json'),cases=prepared,time=time.time(),new_optimizer=0,new_environments=0,new_model_forward=0))
public['PREPARATION_COMPLETE.json']=(Q/'PREPARATION_COMPLETE.json').read_text()
diagnostics={}
for index,(relative,text) in enumerate(sorted(public.items())):
    diagnostics['public_%04d'%index]=dict(relative=relative,text=text,sha256=hashlib.sha256(text.encode()).hexdigest())
print(json.dumps(dict(status='PASS_PREPARED_FROZEN_NOT_LAUNCHED',observed=time.time(),contract_sha256=sha(Q/'CONTRACT.json'),cases=prepared,diagnostics=diagnostics,new_optimizer=0,new_environments=0,new_model_forward=0)))
