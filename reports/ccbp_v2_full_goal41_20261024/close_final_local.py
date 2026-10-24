"""Bind completed evidence and publishable conclusions; no remote execution."""
import json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent
D=R/'final_readonly_20261025_054702'
S=R/'check_readonly_20261025_054624'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_FINAL_AUDIT.json');v=js(D/'EVIDENCE.json');p=js(S/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
assert js(D/'ATTEMPT.json')['exit']==0
assert js(D/'COMPARISON_AND_MECHANISM.json')['status']=='PASS_ACCEPTED_DESCRIPTIVE_ANALYSIS'
assert js(S/'EVALUATION_PROGRESS_AUDIT.json')['status']=='PASS_PROGRESS_NOT_FINAL'
assert v['light']['OUTCOMES.json']['sha256']==p['light']['OUTCOMES.json']['sha256']
assert v['contract']==p['contract_sha256']
assert v['no_residual'] and not v['processes'] and not p['processes']
assert v['light']['controller.exit']['text']=='0'
for arm in 'BCNV':
 assert a['matrix'][arm]['9_9']['episodes']==20
assert a['scientific_main_exploratory_criterion'] is False
status=js(R/'RUN_STATUS.json');old=dict(status)
observed=datetime.datetime.fromtimestamp(v['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
status.update(status='COMPLETE_ACCEPTED_PRIMARY_CRITERION_NOT_MET',observation=observed,observation_unix=v['observed'],formal_episodes=800,phase={'phase':'complete','controller_exit':0,'own_processes':[]},latest_evidence=D.name,final_acceptance='PASS_COMPLETE_LOCAL_ACCEPTANCE',scientific_main_exploratory_criterion=False,automatic_expansion=False,next_check_minutes=None,eta=None,monitor_status='DELETE_AFTER_VERIFIED_PUBLICATION',metrics=a['metrics'],final_success_percent={arm:{str(t):a['matrix'][arm][f'9_{t}']['percent'] for t in range(6,10)} for arm in 'BCNV'})
status['last_capacity_snapshot']=dict(source=S.name,observed=p['observed'],values=p.get('disk',p.get('capacity')),note='Historical completion snapshot, no remaining training or checkpoint-save budget; no cleanup in closure.')
status.pop('capacity',None)
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
active=js(R/'ACTIVE_ROOT.json');active.update(execution_status='COMPLETE_EXIT0',final_acceptance='PASS_COMPLETE_LOCAL_ACCEPTANCE',scientific_main_exploratory_criterion=False,monitor_status='DELETE_AFTER_VERIFIED_PUBLICATION')
(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
intro='''# 最终闭合：2026-10-25 CCBP-v2 完整 GOAL seed41

**四组完整训练、800正式回合及独立验收均已完成；完整v2的主要探索条件未通过，不自动扩展。**

|组别|最终Task6|Task7|Task8|Task9|AUC|FinalAll|
|---|---:|---:|---:|---:|---:|---:|
|B 固定教师蒸馏|35%|100%|85%|55%|76.25%|68.75%|
|C 专家成本监督|30%|100%|85%|90%|83.125%|76.25%|
|N 新路由无锚|30%|100%|85%|75%|78.75%|72.50%|
|V 完整v2软锚|45%|100%|80%|80%|82.50%|76.25%|

V相对C改善Task6保持15pp，但Task8下降5pp、Task9下降10pp；相同FinalAll不能掩盖这一取舍。V的Task8离线执行专家cost/选择regret明显低于C，闭环却较差，证明本轮这些代理改善不足以支持闭环收益结论。没有同历史动作替换，不能归因于具体失败步骤；单seed重复探索，不声称显著或硬保护。

详见[完整结果与机制分析](final_readonly_20261025_054702/FULL_REPORT.md)、[独立最终验收](final_readonly_20261025_054702/LOCAL_FINAL_AUDIT.json)、[闭合证据索引](final_readonly_20261025_054702/CLOSURE_INDEX.json)。全部矩阵、V-C/V-N/V-B配对SF/FS、负单元、梯度尺度、成本和限制均保留。

controller自然exit0且无自身进程。实际验收16个CPU严格100键模型、4份完整恢复、12份本组教师来源/缓存、800初态真实字节及配对noiseSHA、全部1600epoch304400主更新。各Task6 tensor均精确同原ALIGN。额外70隔离更新、正式192和准入48次梯度autograd、教师10820窗口、评估缓存准入及CPU诊断21640窗口688调用单列；训练epoch合计219.441182分钟、评估vector合计71.319672分钟，准备/缓存/保存/工程停顿额外，不称等算力或墙钟总时长。

外置CPU最终wrapper首次实际调用成功。随后本地报告描述器因NumPy int64 JSON序列化失败，已保留原脚本/错误，仅转换描述性计数为Python int并成功复算；0远程命令/前向/更新/环境，无重训重评。早期完整B Task6继承、旧废止分叉以及已授权中间文件清理记录保持。验收完成后仅发布与删除本监控，不再运行实验。

以下为原历史进度，均由本最终段落取代，不代表当前仍在训练或评估。

'''
report=R/'REPORT.md';text=report.read_text(encoding='utf8')
if not text.startswith('# 最终闭合：'):report.write_text(intro+text,encoding='utf8')
(R/'MONITOR_HANDOFF.md').write_text(intro+'\n任务已经闭合。不要再deploy、训练、评估、轮询或扩seed。监控在核验本次远端发布后删除。\n',encoding='utf8')
notes=R/'IMPLEMENTATION_NOTES.md'
with notes.open('a',encoding='utf8') as f:f.write('\n## 最终真实验收\n\nfinal_readonly_20261025_054702实际CPU严格16CP及21640窗口688调用通过；audit_final_local实际完整验收通过。describe_full_final首次因NumPy int64不可JSON序列化失败，wrapper_failure_01保留原源/错误；仅builtin int转换后成功。全部源/输入/科学执行不改，0新增环境或训练。完整报告与CLOSURE_INDEX绑定本次真实证据。\n')
paths=[S/'EVIDENCE.json',S/'LOCAL_AUDIT.json',S/'EVALUATION_PROGRESS_AUDIT.json',R/'AUTHORIZED_CLEANUP_RECEIPT.json',R/'PROTOCOL.md',R/'final_readonly.py',R/'audit_final_local.py',R/'describe_full_final.py',R/'close_final_local.py']
paths += [x for x in D.rglob('*') if x.is_file() and x.name!='CLOSURE_INDEX.json' and '__pycache__' not in x.parts]
bound={str(x.relative_to(R)).replace('\\','/'):dict(bytes=x.stat().st_size,sha256=sha(x)) for x in sorted(paths)}
index=dict(status='PASS_COMPLETE_BOUND_CLOSURE',contract=v['contract'],remote_observed=observed,controller_exit=0,no_residual=True,actual_final_cpu_acceptance=True,local_acceptance=a['status'],scientific_main_exploratory_criterion=False,automatic_expansion=False,evidence_files=bound,publication='Verified separately by PUBLICATION_RECEIPT; this index does not assert a future push',monitor='Delete after verified publication')
(D/'CLOSURE_INDEX.json').write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
assert all(sha(R/n)==m['sha256'] and (R/n).stat().st_size==m['bytes'] for n,m in bound.items())
print(json.dumps(dict(status=index['status'],bound_files=len(bound),observed=observed,primary_criterion=False)))
