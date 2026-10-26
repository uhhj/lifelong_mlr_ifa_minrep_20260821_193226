"""Existing accepted results only: historical RA+ALIGN comparison, no paired-noise claim."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json

R=Path(__file__).resolve().parent
P=R.parent/'20261022_align_object_extension_goal_conditional/final_GOAL_41_20261023_052225'
D=R/'final_readonly_20261026_014654'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=read(P/'LOCAL_FINAL_AUDIT.json');a=read(D/'LOCAL_FINAL_AUDIT.json');v=read(D/'EVIDENCE.json')
assert base['status']=='COMPLETED_LOCAL_INDEPENDENT_PASS' and a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
def metric(r):
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,
 FWT=sum(r[t,t] for t in range(6,10))/4,
 NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
 OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
rr={(x['stage'],x['task']):F(x['current']) for x in base['matrix']}
bm=metric(rr)
assert all(str(bm[k])==x['fraction'] for k,x in base['metrics'].items())
records=json.loads(v['light']['OUTCOMES.json']['text']);bindings=[]
for (stage,task),rate in rr.items():
 p=P/f'current/ra_coreset/evaluation/trials_stage{stage}_task{task}.jsonl'
 rows=[json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s]
 assert len(rows)==20 and sorted(x['initial_state_index'] for x in rows)==list(range(20))
 assert F(sum(x['success'] for x in rows),20)==rate
 for row in rows:
  match=[x for x in records if x['stage']==stage and x['task']==task and x['initial_index']==row['initial_state_index']]
  assert len(match)==3 and all(x['initial_sha256']==row['initial_state_sha256'] and x['initial_file_sha256']==row['initial_file_sha256'] for x in match)
 bindings.append(dict(stage=stage,task=task,sha256=sha(p),initial_bytes_hashes_match=True,has_per_trajectory_noise_hash=all('noise_sha256' in x for x in rows)))
matrix={'RA_ALIGN':{'%d_%d'%k:float(x*100) for k,x in rr.items()}}
metrics={'RA_ALIGN':{k:float(x*100) for k,x in bm.items()}};delta={};negative={}
for arm in 'CDOS':
 ar={(int(k.split('_')[0]),int(k.split('_')[1])):F(x['successes'],20) for k,x in a['matrix'][arm].items()}
 am=metric(ar);assert all(str(am[k])==x['fraction'] for k,x in a['metrics'][arm].items())
 matrix[arm]={'%d_%d'%k:float(x*100) for k,x in ar.items()};metrics[arm]={k:float(x*100) for k,x in am.items()}
 delta[arm]={k:float((am[k]-bm[k])*100) for k in am};negative[arm]={k:matrix[arm][k]-matrix['RA_ALIGN'][k] for k in matrix[arm] if matrix[arm][k]<matrix['RA_ALIGN'][k]}
out=dict(status='PASS_HISTORICAL_DESCRIPTIVE_COMPARISON',source_hashes={'RA_ALIGN_local_acceptance':sha(P/'LOCAL_FINAL_AUDIT.json'),'DOS_local_acceptance':sha(D/'LOCAL_FINAL_AUDIT.json')},
 metrics=metrics,delta_pp_vs_RA_ALIGN=delta,matrix=matrix,negative_cells=negative,initial_record_bindings=bindings,
 paired_noise_verified=False,original_preregistered_C_support_unchanged=a['scientific_exploratory_support'],
 additional_training=0,additional_forwards=0,additional_environments=0,
 limits=['Historical RA+ALIGN is not the auxiliary-expert C arm.','Initial-state file and per-state hashes match, but historical records do not certify matching per-trajectory action noise.','C and DOS Task6 have no new auxiliary loss, yet historical RA+ALIGN scores 80 versus 90; cross-run differences cannot all be assigned to new modules.','DOS use additional training interaction; C/RA+ALIGN are external references, not equal-resource controls.','Post-hoc descriptive baseline comparison does not replace the frozen C comparator or its stopping rule.'])
(R/'RA_ALIGN_BASELINE_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
lines=['# 以历史RA+ALIGN为基线的补充比较','','同一GOAL seed41；只读复用已验收结果，0新训练/前向/环境。原C是RA+ALIGN上再加持续专家监督和margin路由的强对照，不能把C叫作原RA+ALIGN。','','|方法|AUC|相对RA+ALIGN|OldFinal|FinalAll|相对RA+ALIGN|','|---|---:|---:|---:|---:|---:|']
for arm in ['RA_ALIGN','C','D','O','S']:
 m=metrics[arm];dd=delta.get(arm,{'AUC':0,'FinalAll':0});lines.append('|%s|%.4f|%+.4f|%.4f|%.2f|%+.2f|'%(arm,m['AUC'],dd['AUC'],m['OldFinal'],m['FinalAll'],dd['FinalAll']))
lines+=['','|方法|最终Task6|Task7|Task8|Task9|早期7/6|','|---|---:|---:|---:|---:|---:|']
for arm in ['RA_ALIGN','C','D','O','S']:lines.append('|'+arm+'|'+'|'.join(str(matrix[arm][k]) for k in ['9_6','9_7','9_8','9_9','7_6'])+'|')
lines+=['','换成历史RA+ALIGN参考，O和S的AUC及最终平均都更高，D更低。S最终Task8/9为100/95，对历史70/90；但最终Task6为45，对历史55，仍有10pp损失。O最终Task6也低10pp，早期Stage8的Task8低5pp；S的Stage8 Task6/Task8低5/15pp。全部负单元在JSON保留。','', '这是补充描述，不改写预登记判定：相对更强C，O和S没有通过整体门。S最终平均较C高2.5pp，但早期7/6低10pp。','', '已经逐条验证历史200条初态记录与DOS的文件/逐状态SHA相同；历史记录没有逐轨迹noise SHA，不能声称完整随机数配对。Task6没有新增辅助项，历史RA+ALIGN为80%，当前C/D/O/S却均90%，进一步说明跨运行差值不能全归于模块。未为本补充比较重评历史检查点。','',*out['limits']]
(R/'RA_ALIGN_BASELINE_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(metrics=metrics,delta=delta,negative=negative),indent=2))
