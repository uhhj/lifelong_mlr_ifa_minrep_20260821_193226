"""Evidence-bound interpretation of accepted GOAL Q/S, without new science calls."""
from pathlib import Path
import json,sys,hashlib,statistics,datetime
A=Path(__file__).resolve().parent;R=A.parent;D=Path(sys.argv[1]).resolve()
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_FINAL_AUDIT.json');c=js(D/'PRIMARY_S_COMPARISON.json');v=js(D/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_GOAL_LOCAL_ACCEPTANCE' and c['paired_episodes']==600
assert a['evidence_sha256']==sha(D/'EVIDENCE.json') and c['source_Q_final_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json')
ref=c['source_S']['GOAL'];sd=Path(ref['final_path']);sv=js(sd/'EVIDENCE.json');sa=js(sd/'LOCAL_FINAL_AUDIT.json')
assert sha(sd/'EVIDENCE.json')==ref['evidence_sha256'] and sha(sd/'LOCAL_FINAL_AUDIT.json')==ref['audit_sha256']
rows=c['suites']['GOAL'];cells=[(case,k,x) for case,z in c['cases'].items() for k,x in z['cells'].items()]
sf=sum(len(x['Q_fail_S_success']) for _,_,x in cells);fs=sum(len(x['Q_success_S_fail']) for _,_,x in cells)
w=sum(x['S_minus_Q_pp']>0 for _,_,x in cells);l=sum(x['S_minus_Q_pp']<0 for _,_,x in cells)
detail={};stage_rows=[];final_rows=[]
for stage in range(6,10):
 for task in range(6,stage+1):
  q=statistics.mean(c['cases'][f'GOAL_{seed}']['cells'][f'{stage}_{task}']['Q_percent'] for seed in (41,42,43));s=statistics.mean(c['cases'][f'GOAL_{seed}']['cells'][f'{stage}_{task}']['S_percent'] for seed in (41,42,43))
  stage_rows.append(dict(stage=stage,task=task,Q=q,S=s,delta=s-q))
for seed in (41,42,43):
 case=f'GOAL_{seed}';qp=case+'/';sp=f'seed{seed}/'
 qg=json.loads(v['light'][qp+'GRADIENTS_Q.json']['text']);sg=json.loads(sv['light'][sp+'GRADIENTS_S.json']['text'])
 gates=[json.loads(x) for name,entry in sv['light'].items() if name.startswith(sp+'GATES_stage') for x in entry['text'].splitlines()]
 assert len(gates)==6700
 qsupport=json.loads(v['light'][qp+'PROBES_9.json']['text'])['support'];ssupport=json.loads(sv['light'][sp+'PROBES_9.json']['text'])['support']
 detail[case]=dict(Q_last_gradient=qg[-1],S_last_gradient=sg[-1],Q_negative_retain_bc=sum(x['retain_vs_bc']['dot']<0 for x in qg),S_negative_retain_bc=sum(x['retain_vs_bc']['dot']<0 for x in sg),Q_stage9_support=qsupport,S_stage9_support=ssupport,S_gate_blocks=len(gates),S_gate_Q_accepts=sum(x['accepted']=='Q' for x in gates),final_Q=a['cases'][case]['metrics']['FinalAll']['percent'],final_S=c['cases'][case]['S_metrics']['FinalAll'])
 final_rows.append('|%d|%.2f|%.2f|%+.2f|'%(seed,detail[case]['final_Q'],detail[case]['final_S'],detail[case]['final_S']-detail[case]['final_Q']))
assert len(cells)==30 and len(detail)==3
interpret=dict(status='PASS_GOAL_INTERPRETATION_AND_SOURCE_CHECK',source_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),source_comparison_sha256=sha(D/'PRIMARY_S_COMPARISON.json'),wins_ties_losses_S_vs_Q=[w,30-w-l,l],paired_S_success_Q_fail=sf,paired_S_fail_Q_success=fs,by_stage_task_means=stage_rows,training_diagnostics=detail,all_negative_cells=c['all_negative_cells'],scope='GOAL only; OBJECT ongoing',causal_limits=['Different own caches and states; full-path gate ablation, not matched-state single-step causality','Three reused development seeds; no significance or independent confirmation','Upstream pretraining provenance not fully closed','Training gradients and probes do not establish closed-loop gain'])
(D/'INTERPRETATION_AUDIT.json').write_text(json.dumps(interpret,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
lines=['# GOAL Q 对最新 S：独立验收与结果解释','', 'GOAL seed41/42/43 的 Q 已通过独立执行验收，600 回合均与已有最新 S 的初态、噪声、环境 seed、attempt 逐项配对。OBJECT 按原队列继续训练；全六Q实验仍为PARTIAL。S没有重训或重评。','', '验收实际在远端 CPU 严格加载12个阶段CP，检查3个完整恢复状态、12个自身成功缓存及其teacher重放、600条实际初态字节和噪声；本地重新计算采样顺序、单候选预算和全部成功率。GOAL自己的273隔离optimizer/123准入autograd已核；生产228300、训练方向108autograd、120采集另行计账。本次CPU验收0 optimizer、0环境。GOAL完成依据是套件完成标记、三个自身train/evaluate.exit0及无GOAL进程，未伪造整个controller成功。审计前后GOAL轻量文件SHA完全相同。','', '## 三seed均值','', '|指标（%）|Q 无门控|最新 S 有门控|S−Q 百分点|','|---|---:|---:|---:|']
lines+=['|%s|%.4f|%.4f|%+.4f|'%(k,x['Q_mean'],x['S_mean'],x['S_minus_Q_pp']) for k,x in rows.items()]
lines+=['', 'NBT需要单独看：S比Q高%.4f百分点，按该指标反而是Q更好。S的最终旧任务成功率更高，并不等于相对各任务初学水平的下降更小。尤其Task8，S初学成绩更高，随后回落也更大；不能把更高的最终成绩解释成所有遗忘指标均改善。'%rows['NBT']['S_minus_Q_pp']]
lines+=['','NBT越低越好；其余指标越高越好。AUC是按任务再按其后续阶段平均，FWT是新任务对角均值；OldFinal仅最终阶段旧任务6–8，FinalAll为最终四任务平均。','', '## 种子与任务差异','', '|seed|Q 最终均值|S 最终均值|S−Q 百分点|','|---|---:|---:|---:|']+final_rows
lines+=['',f'30个阶段×任务×seed单元中，S相对Q为{w}胜、{30-w-l}平、{l}负。600配对回合中，S成功而Q失败{sf}次，Q成功而S失败{fs}次；同一初态跨阶段/seed重复，不能把600视为600个独立新场景。','', '|阶段/任务|Q三seed均值|S三seed均值|S−Q百分点|','|---|---:|---:|---:|']
lines+=['|%d/%d|%.2f|%.2f|%+.2f|'%(x['stage'],x['task'],x['Q'],x['S'],x['delta']) for x in stage_rows]
lines+=['','Task8是最一致的正向差异：刚学完时S在三个seed分别高10/15/20百分点；到最终阶段，Task8差值为−10/+25/+5，显示这种优势并非每个seed都保持。最终Task6差值+10/0/+20，Task7全部持平，Task9为0/+5/0。因此S的均值优势同时涉及新任务Task8和部分旧任务保持，不能概括为所有任务均改善。seed41最终均值持平，来自Task6改善与Task8退化相互抵消。','', '## 全部反例（S低于Q）','', '|seed|阶段/任务|Q|S|S−Q百分点|Q成功S失败|S成功Q失败|','|---|---|---:|---:|---:|---:|---:|']
lines+=['|%s|%d/%d|%.1f|%.1f|%+.1f|%d|%d|'%(x['case'],x['stage'],x['task'],x['Q_percent'],x['S_percent'],x['S_minus_Q_pp'],len(x['Q_success_S_fail']),len(x['Q_fail_S_success'])) for x in c['all_negative_cells']]
lines+=['','seed41最终Task8退化10百分点；seed43在阶段7的Task6/7、阶段8的Task6各退化5百分点。它们保留在结论中，不重试、过滤或追加实验。逐初态SF/FS索引见PRIMARY_S_COMPARISON.json。','', '## 训练代理与解释边界','', '|seed|Q末次BC / 执行误差|S末次BC / 执行误差|Q/S保持-BC梯度余弦|S接受Q块/总块|','|---|---|---|---|---|']
for seed in (41,42,43):
 x=detail[f'GOAL_{seed}'];q=x['Q_last_gradient'];s=x['S_last_gradient']
 lines.append('|%d|%.4f / %.6f|%.4f / %.6f|%.4f / %.4f|%d/%d|'%(seed,q['bc_new'],q['exec_new'],s['bc_new'],s['exec_new'],q['retain_vs_bc']['cosine'],s['retain_vs_bc']['cosine'],x['S_gate_Q_accepts'],x['S_gate_blocks']))
lines+=['','这些是阶段9第99轮固定训练batch诊断，不是独立验证或整任务总体误差。seed42的Q末次BC与执行误差更低，但闭环最终均值反而低于S；seed41的S末次训练代理更好却最终均值持平。因此不能以训练loss低或单次梯度同向直接预测闭环结果。保持与BC/执行梯度既有同向也有冲突，S的门控仅在固定训练探针和自身状态上进行C/Q选择；接受率不是成功概率。CPU离线旧专家/保持误差已逐标量复算并保留RECOMPUTED文件，也不等价于环境成功率。','', '阶段9自身q（Task6/7/8）如下，来自各自成功采集，内容已随训练路径分化：']
for seed in (41,42,43):
 x=detail[f'GOAL_{seed}'];lines.append('- seed%d：Q %s；S %s。'%(seed,list(x['Q_stage9_support'].values()),list(x['S_stage9_support'].values())))
lines+=['','Q与S各1200接受epoch、228300接受更新、120次训练采集；Q生产物理optimizer调用228300，S为426600（Q少46.48%），自身隔离均273。S在保持阶段多运行一条候选，故本轮不等物理optimizer、FLOPs或墙钟，亦不固定缓存内容。结果支持在本开发协议下完整有门控路径的均值优势，不能单独归因为门控某个条件、解释成共同中间状态单步因果或证明普适显著提升。','', '数据泄露审计没有发现新增训练缓存直接混入正式0–19初态；本次Q也真实核验20–29训练与0–19正式分离。但开发评估集被反复使用，上游预训练数据来源独立性仍未完全闭合。执行验收不等于未触碰测试集上的独立确认。参见leakage_audit_20261031/REPORT.md。','', '证据：本目录LOCAL_FINAL_AUDIT.json、PRIMARY_S_COMPARISON.json、INTERPRETATION_AUDIT.json及原EVIDENCE/diagnostics。未追加训练、闭环评估或科学参数更改。']
report='\n'.join(lines)+'\n'
(D/'GOAL_FINAL_REPORT.md').write_text(report,encoding='utf8');(R/'GOAL_FINAL_REPORT.md').write_text(report,encoding='utf8')
receipt=dict(status='PASS_COMPLETE_GOAL_LOCAL_ACCEPTANCE',scope='GOAL only; OBJECT pending',observed=datetime.datetime.now().astimezone().isoformat(),final_path=str(D),evidence_sha256=sha(D/'EVIDENCE.json'),audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),comparison_sha256=sha(D/'PRIMARY_S_COMPARISON.json'),interpretation_sha256=sha(D/'INTERPRETATION_AUDIT.json'),report_sha256=sha(D/'GOAL_FINAL_REPORT.md'),CPs=12,complete_recoveries=3,cache_teacher_replays=12,formal_episodes=600,main_optimizer=228300,isolated_optimizer=273,acceptance_optimizer=0,acceptance_environments=0,source_S=ref,means=rows)
(R/'GOAL_ACCEPTANCE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8');(D/'ACTUAL_ACCEPTANCE_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(means=rows,win_tie_loss=[w,30-w-l,l],S_success_Q_fail=sf,Q_success_S_fail=fs,per_seed={k:[x['final_Q'],x['final_S']] for k,x in detail.items()}),indent=2))
