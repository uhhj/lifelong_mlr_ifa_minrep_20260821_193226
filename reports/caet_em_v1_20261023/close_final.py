"""Bind completed local audits, all cells, payload proofs and disjoint extra costs."""
import json,hashlib,sys,ast
from pathlib import Path
from fractions import Fraction
R=Path(__file__).resolve().parent;F=Path(sys.argv[1]).resolve();S=Path(sys.argv[2]).resolve();P=Path(sys.argv[3]).resolve()
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def out(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
final=js(F/'LOCAL_FINAL_AUDIT.json');proof=js(P/'EVIDENCE.json');d=js(S/'EVIDENCE.json');M=S/'metadata'
assert final['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and proof['status']=='PASS_CPU_PAYLOAD_BINDING'
assert proof['contract']==final['contract']==d['contract_sha256'] and d['controller_exit']=='0' and not d['own_processes'] and not d['gpu'].strip() and not d['hash_bad']
assert all(v=='0' for v in d['phase_exits'].values())
cpu=R/final['CPU_final'];cp=js(cpu/'EVIDENCE.json');assert cp['status']=='PASS' and cp['B1_B2_Task6_exact']
assert all(proof[k]==0 for k in ('forward_calls','optimizer_calls','environment_steps','server_writes'))
vectors={};mechanism={};matrix={};peaks={}
for a in ('B0','B1','B2'):
 root=M/a;server=js(root/'FINAL_AUDIT.json');assert server['status']=='PASS' and server['metrics_sha256']==sha(root/'METRICS.json')
 assert server['contract']==final['contract'] and server['codec_sha256']==cp['codec_sha256']
 matrix[a]={ast.literal_eval(k):v for k,v in final['arms'][a]['success_counts'].items()}
 peaks[a]={str(t):dict(peak_percent=max(matrix[a][i,t]*5 for i in range(t,10)),final_percent=matrix[a][9,t]*5,decline_pp=max(matrix[a][i,t]*5 for i in range(t,10))-matrix[a][9,t]*5) for t in range(6,10)}
 vectors[a]=[];mechanism[a]={}
 for t in range(6,10):
  expected=proof['arms'][a][str(t)];ds=js(root/'dataset'/('task%d.json'%t));rows=ds['rows'];hist=js(root/'epochs'/('task%d.json'%t))
  assert expected['dataset_sha256']==sha(root/'dataset'/('task%d.json'%t)) and expected['population']==len(rows)
  assert expected['action_valid']==sum(sum(r['action_mask']) for r in rows) and expected['effect_valid']==sum(sum(r['effect_mask']) for r in rows)
  assert expected['sidecar_sha256']==js(root/'sidecars'/('task%d.json'%t))['sha256']
  assert ds==js(M/'B0/dataset'/('task%d.json'%t))==js(M/'B1/dataset'/('task%d.json'%t))
  diag=js(root/'diagnostics'/('task%d.json'%t));assert diag['checkpoint_sha256']==cp['checkpoints'][a][str(t)]['sha256'] and diag['codec_sha256']==cp['codec_sha256']
  assert len(diag['rows'])==len(rows)
  for dr,row in zip(diag['rows'],rows):
   assert all(dr[k]==row[k] for k in ('sample','task','demo','u')) and dr['sample_sha256']==row['sha256']
   assert dr['metrics']['predicted_action']['count']==sum(row['action_mask'])
  if a=='B2' and t>6:
   teacher=js(root/'teachers'/('task%d.json'%t));assert expected['teacher']['sha256']==teacher['sha256'] and expected['teacher']['binding']==teacher['binding']
   assert teacher==js(root/'TRANSFER_ADMISSION_BY_STAGE'/('task%d.json'%t))
   old=sum(sum(r['action_mask']) for r in rows if r['task']<t)
   assert all(h['transfer_anchors']==teacher['accepted'] and h['retained_steps']==old for h in hist)
  else:assert all(h['transfer_anchors']==h['transfer_steps']==h['retained_steps']==h['extra_retention_decoder_windows']==0 for h in hist)
  mechanism[a][str(t)]={k:sum(h[k] for h in hist) for k in ('transfer_anchors','transfer_steps','retained_steps','extra_retention_decoder_windows')}
  for j in range(6,t+1):
   ev=[json.loads(x) for x in (root/'evaluation'/('attempts_stage%d_task%d.jsonl'%(t,j))).read_text().splitlines()]
   ends=[x for x in ev if x['event']=='vector_attempt_complete'];starts=[x for x in ev if x['event']=='vector_attempt_start']
   assert len(ends)==len(starts)==2 and {x['group'] for x in ends}=={0,1}
   vectors[a]+=[end['unix']-next(start['unix'] for start in starts if start['group']==end['group']) for end in ends]
 assert len(vectors[a])==20
cost=js(F/'COST_AUDIT.json');cost['failed_attempts_preserved']=js(R/'FAILED_ATTEMPTS_COST.json')
# Each item is a distinct executed admission. Repeated snapshots are not counted.
admissions={
 'failed_codec03':(R/'FAILED_ATTEMPTS_COST.json',3),
 'codec04':(R/'snapshot_20261023_214425/metadata/shared/ADMISSION_CODEC.json',3),
 'B0_04':(R/'snapshot_20261023_214425/metadata/shared/ADMISSION_B0.json',3),
 'B0_05':(R/'snapshot_20261023_222317/metadata/shared/ADMISSION_B0.json',3),
 'B1_05':(M/'shared/ADMISSION_B1.json',3),
 'B2_05':(R/'snapshot_20261024_031425/metadata/shared/ADMISSION_B2.json',3),
 'B2_recovery06':(M/'shared/ADMISSION_B2.json',3),
 'exact_replay06':(M/'shared/RECOVERY06_ADMISSION.json',2)}
for name,(p,n) in admissions.items():
 value=js(p);assert (value['attempt03']['isolated_optimizer_calls'] if name=='failed_codec03' else value['isolated_optimizer_calls'])==n
cost['disjoint_isolated_optimizer_calls']={k:dict(calls=n,source=str(p.relative_to(R)),sha256=sha(p)) for k,(p,n) in admissions.items()}
cost['total_isolated_optimizer_calls']=sum(n for p,n in admissions.values());assert cost['total_isolated_optimizer_calls']==23
cost['completed_main_updates']=316500;cost['codec_auxiliary_updates']=1900
cost['completed_update_sum_including_isolated']=316500+1900+23
cost['unknown_interrupted_partial_epoch_work']='Additional unknown; not included in the completed update sum.'
cost['formal_vector_minutes']={a:sum(v)/60 for a,v in vectors.items()}
cost['parameters']={a:cp['checkpoints'][a]['6']['parameters'] for a in matrix}
cost['frozen_codec_parameters']=js(M/'shared/CODEC_FREEZE_MANIFEST.json')['parameters']
cost['storage_observation']=dict(source='capacity_readonly_20261024_152446/EVIDENCE.json',sha256=sha(R/'capacity_readonly_20261024_152446/EVIDENCE.json'),scope='Private stable allocated-byte observation before final exit, not total incremental experiment storage; preserved recovery and old assets remain.')
cost['inference_latency']='Not isolated; vector duration includes simulation, policy and environment overhead, and early termination.'
out(F/'COST_AUDIT.json',cost);out(F/'MECHANISM_ACCOUNTING.json',mechanism);out(F/'PEAK_DECLINES.json',peaks)
contrib={}
for a,b in [('B1','B0'),('B2','B1'),('B2','B0')]:
 contrib[a+'-'+b]={str(t):float(sum(Fraction(matrix[a][i,t]-matrix[b][i,t],20) for i in range(t,10))/(10-t)*25) for t in range(6,10)}
 assert abs(sum(contrib[a+'-'+b].values())-final['paired_differences_pp'][a+'-'+b]['AUC'])<1e-10
out(F/'TASK_AUC_CONTRIBUTIONS_PP.json',contrib)
lines=['','完整30单元成功率（每格20回合）：','','|阶段→任务|B0|B1|B2|','|---|---:|---:|---:|']
for t in range(6,10):
 for j in range(6,t+1):lines.append('|%d→%d|'%(t,j)+'|'.join(str(matrix[a][t,j]*5) for a in ('B0','B1','B2'))+'|')
lines+=['','B1−B0的AUC下降8.645833pp：Task6寿命平均贡献−6.5625pp，Task7 +5.416667pp，Task8 −1.25pp，Task9 −6.25pp。Effect接口在本seed未超过匹配直接动作对照，不能称接口已证明有效。',
'','B2−B1的AUC增加4.375pp，其中Task9最终20%→45%贡献+6.25pp，Task8寿命平均下降贡献−1.875pp，Task6/7寿命平均贡献均0。组合帮助了最后新任务，但Task6最终70%→60%、Task8阶段8的95%→75%均为负结果。OldFinal由75%降到73.333333%，不能把NBT较低单独解释为保护成功。T与F未分开消融，增益只能归因组合。',
'','B2仍低于B0：AUC−4.270833pp、OldFinal−3.333333pp、FinalAll−2.5pp。全部负cell见NEGATIVE_CELLS，所有自身峰值下降见PEAK_DECLINES；B2 Task6峰值80%→最终60%，Task7峰值85%→65%。',
'','离线诊断说明接口被使用，但不能替代闭环：B1/B2当前任务的Effect置零动作偏移约0.054–0.127、置换约0.177–0.242；动作证据置换后的误差约0.235–0.276，远大于真实Effect动作误差约0.0046–0.0082；响应或视觉证据置换影响小且方向不一致。支持动作证据在这些诊断中更敏感，不证明视觉/响应完全无用。预测响应误差高于真实Effect重建，动作预测与GT差距也非所有阶段一致；不可把闭环失败唯一归因于固定codec、Temporal或GMM。诊断使用训练/RA样本且四步聚合，有重复探索限制。',
'','成本：每组105500主更新，另共享codec1900、历史及恢复准入合计23隔离更新；未知中断半轮成本另列。B0/B1/B2主epoch计算58.535752/63.497524/72.551298分钟，非总墙钟；额外教师、诊断、证据导出、CPU动作诊断和私有恢复保存不可隐藏为等算力/等总内存。参数分别22946891/22826763/22826763，共享codec4884372。推理纯延迟未单独测量，正式vector耗时不当作纯策略延迟。',
'','完整CPU严格12策略CP（B0每个104键，B1/B2每个117键）、codec最早验证best index96、初态文件字节、B1/B2 Task6 tensor相等、600正式/60vector/30cell、RA载荷与侧车、3阶段真实教师缓存和恢复链均绑定。controller自然exit0且无残留。独立验收是数据与执行有效，不是科学收益为正。',
'','本轮按原授权到此结束；不自动扩seed、追加消融或重试负结果。原位置A失败及所有历史工程失败保留。']
with (F/'FINAL_REPORT.md').open('a',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
sources=[S/'EVIDENCE.json',S/'DOWNLOAD.json',cpu/'EVIDENCE.json',P/'EVIDENCE.json',R/'FAILED_ATTEMPTS_COST.json']+[p for p in F.iterdir() if p.is_file() and p.name!='CLOSURE_INDEX.json']
out(F/'CLOSURE_INDEX.json',dict(status='PASS_COMPLETE_LOCAL_CLOSURE',contract=final['contract'],controller_exit=0,no_processes=True,all_phase_exits_zero=True,formal_episodes=600,vectors=60,cells=30,checkpoints=12,payload_binding_actual=True,scientific_gain_over_B0=False,automatic_seed_extension=False,sources={str(p.relative_to(R)):sha(p) for p in sources}))
print(json.dumps(dict(status='PASS_COMPLETE_LOCAL_CLOSURE',isolated_updates=23,vector_minutes=cost['formal_vector_minutes'],AUC_contributions_pp=contrib),indent=2))
