"""Reconcile actual CPU preparation with manuscript provenance and authorized order."""
from pathlib import Path
import ast,datetime,hashlib,json,sys
R=Path(__file__).resolve().parent
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
snap=Path(sys.argv[1]);e=J(snap/'EVIDENCE.json');receipt=J(R/'PREPARATION_PUBLIC_RECEIPT.json')
assert e['status']=='PASS_PREPARED_FROZEN_NOT_LAUNCHED'
assert e['new_optimizer']==e['new_environments']==e['new_model_forward']==0
c=J(R/'CONTRACT.json');spec=J(R/'SPEC.json');binding=J(R/'MAIN_TABLE_BINDING.json')
assert c['order']==spec['order']==[0,1,2,3,4,5,9,8,7,6]
assert sha(R/'CONTRACT.json')==receipt['contract_sha256']==e['contract_sha256']
assert len(c['cases'])==len(e['cases'])==12
for rel,h in c['files'].items():
    p=R/rel;assert p.is_file() and sha(p)==h,rel
cases={};accepted=physical=0
for case,entry in spec['cases'].items():
    r=R/'cases'/case;inp=J(r/'INPUT.json');p=J(r/'PREPARED_CASE.json');counts=J(r/'INPUT_COUNTS.json')
    assert sha(r/'CONTRACT.json')==c['cases'][case]['contract_sha256']
    assert p==e['cases'][case] and p['status']=='PASS_CPU_INPUT_PREPARATION_NOT_MODEL_OR_GPU_ADMISSION'
    assert not p['cuda_initialized'] and p['new_optimizer']==p['new_environments']==p['new_model_forward']==0
    assert J(r/'PREPARATION_ATTEMPT.json')['exit']==J(r/'IMPORT_ATTEMPT.json')['exit']==0
    assert J(r/'IMPORT_ADMISSION.json')['status']=='PASS_ACTUAL_CPU_IMPORT_NO_SCIENCE'
    expected=([1319,2220,2920,3742],[132,222,292,375]) if inp['suite']=='OBJECT' else ([1969,1889,2326,2879],[197,189,233,288])
    assert [counts['population'][str(s)] for s in range(6,10)]==expected[0]
    assert [p['stages'][str(s)]['batches'] for s in range(6,10)]==expected[1]
    assert p['accepted_budget']==inp['accepted_budget']==sum(expected[1])*100
    assert inp['physical_budget']==(inp['accepted_budget'] if inp['method']=='MLR' else (sum(expected[1])*2-expected[1][0])*100)
    mapping=J(r/'TASK_MAPPING.json')['mapping']
    assert [mapping[str(i)]['original_task'] for i in range(10)]==c['order']
    selection=J(r/'SELECTION_BINDING.json');assert selection['status']=='PASS_ORIGINAL_TASK_MEMBERS_PRESERVED'
    for s in range(6,10):
        key=str(s);ifa=J(r/('EXPECTED_IFA_%d.json'%s))
        assert ifa['info']['task_ids']==list(range(6,s+1)) and ifa['original_task']==c['order'][s]
        assert 0<=counts['old_valid'][key]<=counts['valid'][key]<=counts['population'][key]
        assert selection['tasks'][key]['original_task']==c['order'][s]
    if inp['method']=='MLR':
        provenance=binding['cases'][inp['original_case']]['MLR']
        old=J(provenance['config_path']);new=inp['train_config']
        assert new['policy']==old['policy'] and new['train']==old['train']
        assert new['data']==dict(old['data'],task_order_index=21)
        for key in set(old)-{'feature_dir','buffer_dir','experiment_dir','run_root','contract','data','folder','pretrain','bddl_folder','init_states_folder'}:assert new[key]==old[key],(case,key)
        for name,row in provenance['sources'].items():
            local=r/'source/coreset_exp'/name
            if name!='train.py':assert sha(local)==row['sha256'],(case,name)
        oldtree=ast.parse(Path(provenance['sources']['train.py']['path']).read_text(encoding='utf-8-sig'))
        newtree=ast.parse((r/'source/coreset_exp/train.py').read_text(encoding='utf-8-sig'))
        cls=lambda tree:next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PaperMLR')
        assert ast.dump(cls(oldtree),include_attributes=False)==ast.dump(cls(newtree),include_attributes=False)
    else:
        assert inp['train_config']['policy']==inp['parent']['debr_config']['policy']
        assert inp['train_config']['train']==inp['parent']['debr_config']['train']
        assert inp['train_config']['data']==dict(inp['parent']['debr_config']['data'],task_order_index=21)
        assert len(J(r/'INITIAL_SPLIT.json'))==4
    cases[case]=dict(status='PASS_ACTUAL_CPU_PREPARATION_AND_PROVENANCE',accepted=inp['accepted_budget'],physical=inp['physical_budget'],batches=expected[1],order=mapping,model_or_gpu_admission='NOT_YET_RUN')
    accepted+=inp['accepted_budget'];physical+=inp['physical_budget']
assert accepted==c['accepted']==1156800 and physical==c['production']==1636500
result=dict(status='PASS_PREPARED_NOT_LAUNCHED',observed=datetime.datetime.now().astimezone().isoformat(),evidence=str(snap),contract_sha256=sha(R/'CONTRACT.json'),cases=cases,epochs=4800,accepted=accepted,production_physical=physical,admission_expected=666,formal_expected=2400,training_collection_expected=240,scientific_runs_started=0,new_optimizer=0,new_environment=0,current_queue_protected=True)
(R/'PREPARATION_LOCAL_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(R/'PREPARATION_STATUS.json').write_text(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))
