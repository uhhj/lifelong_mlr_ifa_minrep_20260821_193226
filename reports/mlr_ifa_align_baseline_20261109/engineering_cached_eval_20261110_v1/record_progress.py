"""Record an already-audited snapshot; no remote calls or scientific work."""
from pathlib import Path
import json,sys,datetime,hashlib,statistics
R=Path(__file__).resolve().parent;B=R.parent;D=Path(sys.argv[1]);J=lambda p:json.loads(p.read_text(encoding='utf8'))
e=J(D/'EVIDENCE.json');a=J(D/'LOCAL_AUDIT.json');c=J(D/'CACHED_ENGINEERING_AUDIT.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and c['status']=='PASS_CACHED_ENGINEERING_PROGRESS_NOT_FINAL'
obs=datetime.datetime.fromtimestamp(e['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
L={k:v['text'] for k,v in e['light'].items()};engine=json.loads(L['cached_engineering/STATUS.json']);handoff=c['handoff']
legacy=handoff['legacy_completed_cases'] if handoff else []
timing={}
for case in a['counts']:
 vectors=[]
 for name,text in L.items():
  if not name.startswith(case+'/random/evaluation/attempts_'):continue
  starts={}
  for row in map(json.loads,text.splitlines()):
   if row['event']=='vector_attempt_start':starts[row['group']]=row['unix']
   if row['event']=='vector_attempt_complete':
    vectors.append(dict(stage=row['stage'],task=row['task'],group=row['group'],steps=row['steps'],seconds=row['unix']-starts[row['group']],completed_unix=row['unix']))
 vectors.sort(key=lambda x:x['completed_unix']);full=[v['seconds'] for v in vectors if v['steps']==600]
 if vectors:timing[case]=dict(complete_vectors=len(vectors),all_mean_seconds=statistics.mean(v['seconds'] for v in vectors),full_600_step_count=len(full),full_600_step_mean_seconds=statistics.mean(full) if full else None,vectors=vectors)
status=dict(status='PARTIAL_'+engine['status'],observed=obs,root='/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2',controller=44960,contract=e['contract_sha256'],engineering=R.name,engineering_contract=e['cache_contract_sha256'],previous_engineering_contract=e['engineering_contract_sha256'],counts=a['counts'],epochs=a['epochs'],production_updates=a['production_updates'],formal_episodes=a['formal_episodes'],snapshot=str(D.resolve()),evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),handoff_complete=bool(handoff),handoff=handoff,optimization_deployed=True,optimization_active=bool(c['cache_admissions']),cache_admissions=c['cache_admissions'],combined_environment_mutex=c['combined_environment_mutex'],old_failed_exits_preserved=True,completed_training_or_evaluation_repeated=0,queue='OBJECT600formal then GOAL3train at most2parallel then GOAL600formal',final_wrappers=R.name+' final_readonly/audit_final_local/compare_baseline',final_wrapper_status='STATIC_ONLY_NOT_EXECUTED',cache_tolerance=dict(atol=1e-4,rtol=1e-4),cache_bitwise_equivalence=False,protocol_audit='TWO_ROUND_PROTOCOL_AUDIT_20261110.md')
for name in ['RUN_STATUS.json','ACTIVE_ROOT.json']:(B/name).write_text(json.dumps(status,indent=2)+'\n',encoding='utf8')
tim=dict(status='OBSERVED_VECTOR_TIMING_NOT_CLOSED_LOOP_EQUIVALENCE',observed=obs,cases=timing,limits=['Full-600-step comparisons match step counts, not an identical trajectory or same checkpoint A/B.','Early success vectors must not be compared to full600 vectors as an implementation speedup.','No stable whole-suite speed multiplier or final success guarantee is inferred.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(tim,indent=2)+'\n',encoding='utf8')
rows=['# 已审计的缓存工程进度','',obs+'，'+status['status']+'。','', '|case|完整训练轮|正式/200|','|---|---:|---:|']
rows += [f"|{case}|{v['epochs']}|{v['formal_episodes']}|" for case,v in a['counts'].items()]
rows += ['',f"自然交接：{bool(handoff)}；实际缓存准入：{c['cache_admissions']}；合并环境锁：{c['combined_environment_mutex']}。",'','科学合同及原模型/前缀验证通过。缓存容差1e-4，保留原global Normal与done处理，不声称逐位闭环等价。最终CPU仍待完整队列自然结束。']
(D/'PROGRESS_REPORT.md').write_text('\n'.join(rows)+'\n',encoding='utf8')
paragraph='\n'.join(rows[1:])+f'\n\n证据 {R.name}/{D.name}。继续使用本cached工程check/audit_progress/record_progress；最终也用本工程final_readonly/audit_final_local/compare_baseline。不得重复controller、部署、已完成轮或正式回合。\n\n'
p=B/'MONITOR_HANDOFF.md';p.write_text('## '+obs+' 实际复查\n'+paragraph+p.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(dict(status=status['status'],observed=obs,handoff_complete=bool(handoff),formal_episodes=a['formal_episodes'],cache_admissions=c['cache_admissions'])))
