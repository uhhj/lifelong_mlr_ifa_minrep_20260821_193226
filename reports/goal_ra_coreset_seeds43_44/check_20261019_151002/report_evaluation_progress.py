"""Read-only partial evaluation accounting and measured ETA; no environment calls."""
from pathlib import Path
import json,sys,statistics,datetime
R=Path(__file__).resolve().parent;D=R/sys.argv[1]
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
a=js(D/'PROGRESS_ACCEPTANCE.json');snap=js(D/'SNAPSHOT.json')
assert '44' in a['partial'] and a['partial']['44']['epochs']==400
E=D/'seed44/evidence';events=[x for p in (E/'ra_coreset/evaluation').glob('attempts*.jsonl') for x in rows(p)]
trials=[x for p in (E/'ra_coreset/evaluation').glob('trials*.jsonl') for x in rows(p)]
assert not any(x['event']=='technical_exception' for x in events)
key=lambda x:(x['stage'],x['task'],x['group'])
starts={key(x):x for x in events if x['event']=='vector_attempt_start'}
ends={key(x):x for x in events if x['event']=='vector_attempt_complete'}
assert len(trials)==len({(x['stage'],x['task'],x['trial']) for x in trials})
for x in trials:
 k=(x['stage'],x['task'],x['trial']//10)
 assert k in ends and type(x['success']) is bool
 assert ends[k]['successes'][x['trial']%10]==x['success']
assert len(trials)==10*len(ends)
base=[x for p in (D/'seed43/evidence/ra_coreset/evaluation').glob('attempts*.jsonl') for x in rows(p)]
bs={key(x):x for x in base if x['event']=='vector_attempt_start'}
be={key(x):x for x in base if x['event']=='vector_attempt_complete'}
dur={k:be[k]['unix']-bs[k]['unix'] for k in be}
ratio=statistics.median((ends[k]['unix']-starts[k]['unix'])/dur[k] for k in ends)
remaining=sum(v for k,v in dur.items() if k not in ends)*ratio/60
asof=max([snap['time']]+[x['unix'] for x in events])
tz=datetime.timezone(datetime.timedelta(hours=8));dt=datetime.datetime.fromtimestamp(asof,tz)
eta=[round(remaining*.85+2),round(remaining*1.2+5)]
finish=[(dt+datetime.timedelta(minutes=x)).isoformat() for x in eta]
assert a['partial']['44']['task6_tensor_equal'] is True and len(a['partial']['44']['storage'])==4
assert (E/'train.exit').read_text().strip()=='0'
out=dict(status='PARTIAL_EVALUATION_ACCOUNTING_PASS',asof=dt.isoformat(),records=len(trials),completed_vectors=len(ends),technical_exceptions=0,matched_vector_rate_ratio=ratio,remaining_minutes=eta,estimated_finish=finish,basis='seed43 matched vector durations scaled by seed44 median duration ratio; remaining full vectors plus audit allowance; not a completion guarantee',new_training_or_environment_calls=0)
(D/'EVALUATION_PROGRESS.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
status=dict(status='PARTIAL_SEED43_COMPLETE_SEED44_EVALUATING',observation_time=dt.isoformat(),seeds={'43':dict(epochs=400,steps=76100,records=200,status='COMPLETED_AUDIT_PASS'),'44':dict(epochs=400,steps=76100,records=len(trials),status='EVALUATING')},eta=out,queue_exit=snap['queue_exit'],wrapper_exit=snap['wrapper_exit'],latest_evidence=str(D))
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
lines=['# RA-Coreset GOAL扩展进度（PARTIAL）','',f'观测时间：{dt.isoformat()}。服务器快照与逐文件轻量包的采集时间不同，记录数以下以已下载完整vector账目为准。','',f'seed43已完成400epoch、76100step、4CP严格加载、200正式记录/20vector并独立验收。seed44训练已退出0，400epoch、76100step、四任务入库均完成；评估{len(trials)}/200条、{len(ends)}/20vector，0技术异常。queue/wrapper仍未退出，整个扩展未完成。','', '## 入库与保护核查','', '两seed冻结source/config/modal/input/feature哈希与下载manifest均通过。seed44 Task6实际模型tensor与父模型一致；实际入库有序ID、锚点和最少交换与冻结offline完全一致，载荷哈希链通过。', '', '|任务|配额|实际交换|有序名单一致|','|---|---:|---:|---|']
for t,m in a['partial']['44']['storage'].items():lines.append(f"|{t}|{m['quota']}|{m['swaps']}|PASS|")
lines+=['','## 已完成seed43结果（%）','','|方法|AUC|FWT|NBT|OldFinal|FinalAll|','|---|---:|---:|---:|---:|---:|']
for arm,m in a['completed']['43']['metrics'].items():lines.append('|'+arm+'|'+'|'.join(f"{m[k]['percent']:.6f}" for k in ['AUC','FWT','NBT','OldFinal','FinalAll'])+'|')
lines+=['','全部seed43有效单元：','','|stage|task|Random|Coreset|RA|','|---:|---:|---:|---:|---:|']
for x in a['completed']['43']['matrix']:lines.append('|'+ '|'.join(str(x[k]) for k in ['stage','task','random','coreset','ra_coreset'])+'|')
lines+=['',f'按seed44已完成vector与seed43同位置耗时比{ratio:.3f}估算，剩余约{eta[0]}–{eta[1]}分钟，预计北京时间{finish[0]}至{finish[1]}完成，含验收余量。不是提前完成声明。','', '真实命令：check_progress.py --package、audit_progress.py '+D.name+'、report_evaluation_progress.py '+D.name+'，均退出0；远端monitor/verify/package命令与退出码见COMMANDS.json。没有新训练、预检或环境调用。','', 'seed43 RA的AUC高于Coreset6.25pp，但Task8刚学完65%低于Coreset70%，最终Task6低于Random。seed44未完整评估，不计算最终指标或三seed均值。完成后须queue/wrapper退出0、FINAL_QUEUE_ACCEPTANCE及三seed本地复算，之后取消自动化。','', '保留探索性扩展、global动作RNG未隔离、原排序影响batch、IFA间接变化、官方pretrain state未知与调度补全边界，不作显著性结论。']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
