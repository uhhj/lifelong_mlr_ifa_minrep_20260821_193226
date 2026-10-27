"""Close local experiment records only after actual independent acceptance."""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);CAP=Path(sys.argv[2])
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(D/'LOCAL_FINAL_AUDIT.json');v=js(D/'EVIDENCE.json');comp=js(D/'COMPARISON.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and comp['final_audit_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json')
matched=js(D/'MATCHED_DIAGNOSTIC_ANALYSIS.json')
assert matched['final_audit_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json')
assert (D/'MECHANISM_ANALYSIS.md').stat().st_size>1000
assert not v['processes'] and v['light']['controller.exit']['text']=='0'
previous=js(R/'check_readonly_20261026_210507/EVIDENCE.json')
prefix_counts={}
for seed in (41,42,43):
 name=f'seed{seed}/EPOCHS_S.json'
 p=json.loads(previous['light'][name]['text']);q=json.loads(v['light'][name]['text'])
 assert q[:len(p)]==p;prefix_counts[str(seed)]=len(p)
 assert v['light'][f'seed{seed}/train_S.exit']['text']==v['light'][f'seed{seed}/evaluate.exit']['text']=='0'
 for t in range(6,10):
  name=f'seed{seed}/STAGE_S_{t}.json'
  if name in previous['light']:assert previous['light'][name]['sha256']==v['light'][name]['sha256']
c=js(CAP/'EVIDENCE.json')
for mount,row in c['capacity'].items():
 assert row['remaining']==max(0,(6*1024**3 if mount=='/data' else 0)-row['allocated_unique'])
 assert row['reserve']==(2 if mount=='/data' else 1)*1024**3
 assert row['headroom']==row['free']-row['remaining']-row['reserve']>=0
assert all(c[k]==0 for k in ['payload_reads','server_writes','forwards','optimizer_updates','environments'])
save(CAP/'LOCAL_AUDIT.json',dict(status='PASS_NOT_FINAL_CAPACITY_ONLY',observation=c['observed'],capacity=c['capacity'],evidence_sha256=sha(CAP/'EVIDENCE.json')))
stamp=lambda t:dt.datetime.fromtimestamp(t,dt.timezone(dt.timedelta(hours=8))).isoformat()
finished=stamp(json.loads(v['light']['STATUS.json']['text'])['time']);observed=stamp(v['observed'])
save(D/'CLOSURE_AUDIT.json',dict(status='PASS_COMPLETE_BINDING',contract=a['contract'],controller_exit=0,no_residual=True,
 completed_at=finished,accepted_observation=observed,previous_epoch_prefixes_exact=prefix_counts,
 final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),comparison_sha256=sha(D/'COMPARISON.json'),
 matched_analysis_sha256=sha(D/'MATCHED_DIAGNOSTIC_ANALYSIS.json'),mechanism_analysis_sha256=sha(D/'MECHANISM_ANALYSIS.md'),
 capacity_audit_sha256=sha(CAP/'LOCAL_AUDIT.json'),monitor_removal_after_publication=True))
s=js(R/'RUN_STATUS.json');progress=js(D/'LOCAL_AUDIT.json')
s.update(status='COMPLETE_ACCEPTED',scientific_status='FINITE_EXPERIMENT_ACCEPTED_NO_SIGNIFICANCE_CLAIM',
 observation=observed,completed_at=finished,latest_evidence=D.name,final_audit='LOCAL_FINAL_AUDIT.json',
 seeds={seed:dict(counts=x['counts']['S'],admissions=x['admissions'],memory=x['memory'],formal_episodes=x['formal_episodes']) for seed,x in progress['seeds'].items()},
 main_updates=316500,training_collection_episodes=120,formal_episodes=600,isolated_optimizer_updates=66,
 next_check_minutes=None,eta=None,phase=dict(phase='complete',controller_exit=0,no_residual=True),
 capacity_evidence=CAP.name,metrics={s:x['metrics'] for s,x in a['seeds'].items()},
 three_seed_statistics=comp['three_seed_statistics'],monitor_status='DELETE_AFTER_VERIFIED_PUBLICATION')
save(R/'RUN_STATUS.json',s)
active=js(R/'ACTIVE_ROOT.json');active.update(status='COMPLETE_ACCEPTED_NO_ACTIVE_PROCESS',last_observation=observed,last_evidence=D.name,controller_exit=0,completed_at=finished);save(R/'ACTIVE_ROOT.json',active)
history=R/'PROGRESS_HISTORY_THROUGH_20261026_2107.md'
if not history.exists():history.write_bytes((R/'REPORT.md').read_bytes())
summary=f'''# S OBJECT seed41/42/43：已完成并通过独立验收

服务器完成时间：{finished}。本地最终接受观察：{observed}。
三seed1200完整epoch、316500主更新、120训练采集、600正式/60vector/30cell全部完成；控制器、训练、采集和评估自然exit0，无自身进程。12个CPU strict检查点、3完整恢复、12缓存源教师与抽样、初态真实字节及配对噪声、全部标量实际验收通过，不再以静态脚本检查代替接受。

三seed平均AUC85.0347%、OldFinal82.2222%、FinalAll82.50%；历史RA+ALIGN分别80.1736%、74.4444%、75.8333%。历史对照非同动作噪声或等交互，不称独立因果或显著。必须保留seed41 Task9=55%及seed43最终Task6=20%的负结果。正式收益与训练缓存成功率分开。

最终证据目录：{D.name}。完整表格及负单元见下文；机制解释见该目录MECHANISM_ANALYSIS.md，共同窗口统一尺度复算见MATCHED_DIAGNOSTIC_ANALYSIS.json。历史进度保存在PROGRESS_HISTORY_THROUGH_20261026_2107.md，之前的包装错误、部署与准入证据不删除。

本轮只重启已经退出的本机凭据worker（新PID332），从用户已授权Credential Manager读入内存；没有重启服务器训练/控制器/评估。之前的本地进度停留在21:05，今天新观察证明远端已于昨晚自然完成，不补造中间监控记录。最终CPU核验新增0优化器/0环境；独立CPU诊断成本单列。发布完成后删除监控，不自动扩seed、补采或扫描。

'''
(R/'REPORT.md').write_text(summary+(D/'REPORT.md').read_text(encoding='utf8')+'\n'+(D/'MECHANISM_ANALYSIS.md').read_text(encoding='utf8'),encoding='utf8')
(R/'MONITOR_HANDOFF.md').write_text(summary+'\n全部有限预算、分析、独立接受均已完成。仅待发布核远端SHA后删除s-object-seed41-42-43；不得启动任何新实验。\n',encoding='utf8')
print(json.dumps(dict(status=s['status'],completed_at=finished,observation=observed,statistics=comp['three_seed_statistics']),indent=2))
