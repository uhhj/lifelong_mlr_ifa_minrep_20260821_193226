"""Bind completed local acceptance and analysis; never contacts the experiment server."""
import datetime
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parent
D=R/'final_readonly_20261025_181226'
S=R/'check_readonly_20261025_181206'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(D/'LOCAL_FINAL_AUDIT.json');v=read(D/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['scientific_main_exploratory_criterion'] is False
assert v['no_residual'] and not v['processes'] and v['environment_calls']==v['optimizer_updates']==0
assert all(v['light'][k]['text'].strip()=='0' for k in ['controller.exit','evaluate.exit','train_C.exit','train_R.exit','train_G.exit'])
assert read(D/'MECHANISM_DIAGNOSTICS.json')['acceptance_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json')
now=datetime.datetime.now().astimezone().isoformat()
observation=datetime.datetime.fromtimestamp(v['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
status=read(R/'RUN_STATUS.json')
status.update(status='COMPLETE_ACCEPTED_SCIENTIFIC_CRITERION_NOT_MET',observation=observation,
              completed_acceptance=now,latest_evidence=D.name,formal_episodes=600,
              final_wrappers='ACTUAL_REMOTE_CPU_AND_LOCAL_ACCEPTANCE_PASS',
              scientific_main_exploratory_criterion=False,no_automatic_expansion=True,
              phase={'phase':'complete','controller':2177,'natural_exit':0,'residual_processes':0},
              next_check_minutes=None,next_check_basis='Finished; delete monitor after verified publication',
              monitoring='DELETE_AFTER_VERIFIED_PUBLICATION',final_metrics=a['metrics'],
              final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'))
status['fresh_remote_capacity']={'observation':read(S/'EVIDENCE.json')['observed'],
    'source':S.name+'/EVIDENCE.json',
    'capacity':read(S/'EVIDENCE.json')['capacity'],
    'note':'Completion observed; earlier full peak reservation retained only as historical evidence.'}
write(R/'RUN_STATUS.json',status)
active=read(R/'ACTIVE_ROOT.json');active.update(status=status['status'],observed=observation,natural_exit=0,residual_processes=0,final_evidence=D.name)
write(R/'ACTIVE_ROOT.json',active)
summary='''# 最终接受完成：CCBP-v3 GOAL seed41

C/R/G完整1200epoch、228300主更新、600正式回合均自然完成。实际12CP CPU严格加载、3完整恢复态、9教师缓存源、真实初态字节与配对noise，以及本地全部结果/标量独立验收PASS。科学主条件未通过，停止本版本，不自动扩seed或调参。

|组别|最终Task6|Task7|Task8|Task9|AUC|OldFinal|FinalAll|
|---|---:|---:|---:|---:|---:|---:|---:|
|C|40%|100%|100%|90%|87.5%|80%|82.5%|
|R|35%|100%|60%|80%|77.8125%|65%|68.75%|
|G|50%|100%|80%|85%|83.125%|76.6667%|78.75%|

G对C最终Task6 +10pp，但Task8 −20pp、Task9 −5pp；OldFinal、Task8、Task9三项主条件均未满足。Task7之后G的Task6离线cost和路由regret更低，闭环却40%对C80%；最终Task8离线较好的R也只有60%。因此不能用代理改善宣布保持有效，不能归因于某个未做干预验证的单步物理失败。

完整报告：[最终机制分析](final_readonly_20261025_181226/ANALYSIS.md)、[全矩阵及SF/FS](final_readonly_20261025_181226/RESULT_TABLES.md)、[独立最终验收](final_readonly_20261025_181226/LOCAL_FINAL_AUDIT.json)、[闭合SHA索引](final_readonly_20261025_181226/CLOSURE_INDEX.json)。历史进度均保留下方，其PARTIAL不再代表当前状态。验收成功不等于方法改进成功。

'''
old=(R/'REPORT.md').read_text(encoding='utf8')
if not old.startswith('# 最终接受完成：'):
    (R/'REPORT.md').write_text(summary+'---\n\n'+old,encoding='utf8')
(R/'MONITOR_HANDOFF.md').write_text(summary+"\n本实验已经闭合。不要再次deploy、训练、评估或final诊断；仅完成验证发布及删除ccbp-v3-goal41监控。所有旧失败、负单元与成本均保留。新研究必须等待新的明确用户请求。\n",encoding='utf8')
notes='\n\n## 最终实际验收 '+now+'\n最终final_readonly于18:12启动、18:16完成；CPU12CP/3恢复/9教师/600初态字节与noise实际通过，本地独立audit和共同窗口逐身份SHA通过。新describe_final仅聚合已验收标量、close_final只绑定本地证据，0额外训练或环境。科学主条件False，按预登记停止v3。原进度/失败保留。\n'
with (R/'IMPLEMENTATION_NOTES.md').open('a',encoding='utf8') as f:f.write(notes)
targets=[p for p in D.rglob('*') if p.is_file() and p.name!='CLOSURE_INDEX.json' and 'light' not in p.relative_to(D).parts]
targets += [R/n for n in ['PROTOCOL.md','IMPLEMENTATION_NOTES.md','REPORT.md','RUN_STATUS.json','ACTIVE_ROOT.json','MONITOR_HANDOFF.md','final_readonly.py','audit_final_local.py','describe_final.py','close_final.py']]
targets += [S/n for n in ['EVIDENCE.json','LOCAL_AUDIT.json','EVALUATION_PROGRESS_AUDIT.json']]
assert all(p.suffix in {'.py','.json','.md'} for p in targets)
bound={p.relative_to(R).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(targets))}
index=dict(status='COMPLETE_ACCEPTED_SCIENTIFIC_CRITERION_NOT_MET',created=now,
           contract=v['contract_sha256'],final_directory=D.name,formal_epochs=1200,formal_updates=228300,
           formal_episodes=600,strict_cpu_checkpoints=12,scientific_criterion=False,
           no_remaining_experiment_work=True,no_automatic_expansion=True,files=bound,
           publication='Commit and remote verification recorded in local PUBLICATION_RECEIPT files; not self-referential')
write(D/'CLOSURE_INDEX.json',index)
assert all(sha(R/n)==x['sha256'] for n,x in index['files'].items())
print(json.dumps(dict(status=index['status'],bound_files=len(bound),observation=observation),indent=2))
