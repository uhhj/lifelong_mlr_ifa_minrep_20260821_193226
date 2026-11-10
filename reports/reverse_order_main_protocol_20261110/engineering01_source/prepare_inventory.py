"""Bind the exact manuscript rows and prepare a metadata-only remote request."""
from pathlib import Path
import hashlib,json,datetime
R=Path(__file__).resolve().parent
W=R.parents[1]
P=R.parent
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mlr=J(P/'20261107_mlr_ifa_reproduction_audit/AUDIT.json')
debr=J(P/'20261110_s_stripped_mlr_ifa_align/FULL_S_BASELINE_BINDING.json')
requests={};sources={}
for suite in ['OBJECT','GOAL']:
    for seed in [41,42,43]:
        key=suite+'_'+str(seed)
        parent=J(P/'20261110_s_stripped_mlr_ifa_align/cases'/(suite+'_MLR_'+str(seed))/'INPUT.json')
        historical=mlr['results'][key]
        light=P/historical['root']
        modal=J(light/'MODAL_CONTRACT.json')
        cfg=light/'resolved_random.json'
        if not cfg.exists():cfg=P/'20261016_seed43_three_arm_repeat_v1/launch_evidence/rc/resolved_random.json'
        ec=light/'random/evaluation/resolved_actual.json'
        main_sources={}
        for name,relative in historical['source_binding'].items():
            path=W/relative
            assert sha(path)==modal['source_sha256']['coreset_exp/'+name]
            main_sources[name]=dict(path=str(path),sha256=sha(path))
        source_root=(W/historical['source_binding']['train.py']).parents[1]
        sources[key]=dict(MLR=dict(actual_result_root=str(light),main_metric=historical['metrics'],config_path=str(cfg),config_sha256=sha(cfg),eval_config_path=str(ec),eval_config_sha256=sha(ec),sources=main_sources,source_root=str(source_root),modal_contract_path=str(light/'MODAL_CONTRACT.json'),modal_contract_sha256=sha(light/'MODAL_CONTRACT.json')),
                          DEBR=dict(parent_directory=str(P/('20261028_support_gate_full_s_object' if suite=='OBJECT' else '20261029_support_gate_full_s_goal')/('seed'+str(seed))),accepted_result=debr['cases'][key],suite_acceptance=debr['suites'][suite]))
        requests[key]=dict(suite=suite,seed=seed,main_mlr_parent=parent['parent'],debr_align_parent=parent['align_parent'],debr_align_source=parent['align_source'],labels=parent['labels'],official_pretrain_sha256=historical['pretrain_sha256'])
binding=dict(status='BOUND_MAIN_TABLE_SOURCES_NOT_NEW_EXECUTION',observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),order=[0,1,2,3,4,5,9,8,7,6],incremental_original_task_ids=[9,8,7,6],pretraining_original_task_ids=list(range(6)),main_summary=dict(MLR=mlr['summary'],DEBR={s:debr['suites'][s]['mean_sample_sd'] for s in ['OBJECT','GOAL']}),cases=sources)
for name,obj in [('MAIN_TABLE_BINDING.json',binding),('REQUESTED_PARENTS.json',requests)]:
    (R/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
template=(R/'preflight_template.py').read_text(encoding='utf-8')
(R/'preflight_readonly.py').write_text('PARENTS='+ascii(requests)+'\n'+template,encoding='utf-8')
print(json.dumps(dict(status=binding['status'],cases=len(sources),main_summary=binding['main_summary']),ensure_ascii=False))
