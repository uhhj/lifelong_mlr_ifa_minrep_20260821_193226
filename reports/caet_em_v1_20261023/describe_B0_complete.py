"""Recompute completed B0 outcomes locally; no training, environment or partial metrics."""
import json,hashlib,datetime
from pathlib import Path
from fractions import Fraction as F
R=Path(__file__).resolve().parent
S=R/'snapshot_20261024_132735';B=S/'metadata/B0'
A=R.parent/'20261022_caet_v3_tac_eftr_object41/final_ALIGN_seed41_20261022_220252/LOCAL_FINAL_AUDIT.json'
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=js(S/'EVALUATION_PROGRESS_AUDIT.json')
assert audit['arm']=='B0' and audit['complete_episodes']==200 and audit['complete_vectors']==20 and audit['closed_complete_cells']==10 and audit['technical_exceptions']==0
download=js(S/'DOWNLOAD.json');matrix={};sources={}
for s in range(6,10):
 for t in range(6,s+1):
  p=B/'evaluation'/('trials_stage%d_task%d.jsonl'%(s,t));rel=p.relative_to(S/'metadata').as_posix()
  assert sha(p)==download[rel]['sha256']==download[rel]['at_snapshot']['sha256'];sources[rel]=sha(p)
  rows=[json.loads(x) for x in p.read_text().splitlines()]
  assert len(rows)==20 and [x['initial_state_index'] for x in rows]==list(range(20))
  matrix[s,t]=F(sum(x['success'] for x in rows),20)
def metrics(m):
 return dict(AUC=sum(sum(m[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(m[t,t] for t in range(6,10))/4,NBT=sum(sum(m[t,t]-m[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(m[9,t] for t in range(6,9))/3,FinalAll=sum(m[9,t] for t in range(6,10))/4)
values=metrics(matrix);server=js(B/'METRICS.json')
assert all(F(server['metrics'][k]['fraction'])==v for k,v in values.items())
old=js(A);references={};negative={};differences={}
for name,key,mkey in [('ALIGN','metrics','current'),('RA','ra','ra')]:
 m={(x['stage'],x['task']):F(x[mkey]) for x in old['matrix']};v=metrics(m)
 assert all(v[k]==F(old[key][k]['fraction']) for k in v)
 references[name]={k:float(x*100) for k,x in v.items()}
 differences[name]={k:float((values[k]-v[k])*100) for k in v}
 negative[name]={str(k):float((matrix[k]-m[k])*100) for k in matrix if matrix[k]<m[k]}
preflight=R/'recovery06_preflight_readonly_20261024_132349/EVIDENCE.json';pre=js(preflight)
for t in range(6,10):assert pre['checkpoints']['B0_task%d'%t]['sha256']==js(B/'TRAINING_COMPLETE.json')['checkpoints'][str(t)]
report=dict(status='PASS_COMPLETE_B0_RECORD_RECOMPUTATION_NOT_WHOLE_QUEUE_FINAL',time=datetime.datetime.now().astimezone().isoformat(),snapshot=S.name,arm='B0_READOUT_BC4',seed=41,suite='OBJECT',episodes=200,vectors=20,cells=10,total_successes=sum(int(v*20) for v in matrix.values()),metrics={k:dict(fraction=str(v),percent=float(v*100)) for k,v in values.items()},matrix=[dict(stage=s,task=t,successes=int(v*20),episodes=20,percent=float(v*100)) for (s,t),v in matrix.items()],external_reference_metrics=references,differences_pp=differences,negative_cells_pp=negative,source_trial_sha256=sources,reference_audit_sha256=sha(A),evaluation_audit_sha256=sha(S/'EVALUATION_PROGRESS_AUDIT.json'),CPU_checkpoint_preflight_sha256=sha(preflight),full_B0_CPU_checkpoint_preflight=True,independent_initial_state_file_bytes='PENDING_WHOLE_QUEUE_FINAL',whole_queue_acceptance='PENDING',single_seed_exploratory=True,causal_limit='B0 changes both control readout and four-step supervision/head relative to ALIGN; no single-factor causal attribution.')
(R/'B0_COMPLETE_RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
lines=['# B0 OBJECT seed41 已完成闭环结果','','200正式episode、20vector、10cell已完整，从原记录独立复算；当前为完整B0记录结果，不冒充全队列最终验收。四个B0完整CP已在recovery06 CPU预检严格核验；独立初态文件字节及全部三组最终验收仍待完成。','','|指标|B0|RA41（外部参考）|ALIGN41（外部参考）|','|---|---:|---:|---:|']
for k,v in values.items():lines.append('|%s|%.6f|%.6f|%.6f|'%(k,float(v*100),references['RA'][k],references['ALIGN'][k]))
lines+=['','指标单位为%；NBT低好。200条直接合计146次成功（73%），与按任务等权的AUC69.6875%不是同一汇总方式。','','|训练结束阶段|Task6|Task7|Task8|Task9|','|---|---:|---:|---:|---:|']
for s in range(6,10):lines.append('|Task%d|'%s+'|'.join('%.0f%% (%d/20)'%(float(matrix[s,t]*100),int(matrix[s,t]*20)) if t<=s else '—' for t in range(6,10))+'|')
lines+=['','总体AUC低于RA 3.333333pp、低于ALIGN 11.770833pp。主要短板在新Task7刚学完45%及Task9最终45%；Task8最终100%，旧Task平均76.666667%。旧任务平均改善不能掩盖Task6从中间峰值95%降至65%（30pp）。NBT为负代表相对刚学完基线的平均后向改善，不代表所有任务都不遗忘。','','B0没有效果接口/T/F，不可将这些结果当作B1/B2模块结论。相对旧ALIGN同时改变控制token读出、四步监督及动作头，不能将下降单独归因于最新帧读出。单seed探索；全部负单元保存在B0_COMPLETE_RESULT.json。']
(R/'B0_COMPLETE_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(metrics=report['metrics'],differences_pp=differences,successes=report['total_successes'],status=report['status']),indent=2))
