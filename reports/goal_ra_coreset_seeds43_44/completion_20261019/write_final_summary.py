"""Render verified three-seed results, all tasks and limitations without reruns."""
from pathlib import Path
import json,sys,datetime
R=Path(__file__).resolve().parent;D=R/sys.argv[1];a=json.loads((D/'LOCAL_FINAL_ACCEPTANCE.json').read_text());assert a['status']=='COMPLETED'
keys=['AUC','FWT','NBT','OldFinal','FinalAll'];arms=['random','coreset','ra_coreset'];labels={'random':'Random','coreset':'Coreset','ra_coreset':'RA-Coreset'}
lines=['# RA-Coreset GOAL 三seed最终报告','', '状态：COMPLETED。本地独立复算PASS，服务器FINAL_QUEUE_ACCEPTANCE PASS；queue、wrapper及各seed训练/评估/验收退出0，无残留训练或评估进程。','', '三seed合计1200完整epoch、228300 optimizer step、12个ep99模型严格加载凭证、600条正式闭环、60个vector attempts、0技术异常。原Random/Coreset共1200条历史记录仅只读复用，没有重训或重评。权重未下载；严格加载是服务器实际执行、本地校验凭证。','', '## 逐seed结果（%）','','|Seed|方法|AUC|FWT|NBT|OldFinal|FinalAll|','|---:|---|---:|---:|---:|---:|---:|']
for seed,v in a['per_seed'].items():
 for arm in arms:lines.append('|'+seed+'|'+labels[arm]+'|'+'|'.join(f"{v['metrics'][arm][k]['percent']:.6f}" for k in keys)+'|')
lines+=['','## 三seed均值 ± 样本标准差（ddof=1，百分点）','','|方法|AUC|FWT|NBT|OldFinal|FinalAll|','|---|---:|---:|---:|---:|---:|']
for arm in arms+['RA_minus_Coreset','RA_minus_Random']:
 lines.append('|'+labels.get(arm,arm)+'|'+'|'.join(f"{a['statistics'][arm][k]['mean_percent']:.6f} ± {a['statistics'][arm][k]['sample_sd_pp']:.6f}" for k in keys)+'|')
lines+=['','差值的SD直接由三个逐seed配对差计算，不是两组SD相减。NBT为遗忘量，越低越好；负数代表相对初学后出现净改善。AUC按每任务各后续阶段均值再对任务平均，不能把200条直接平均替代。精确分数及公式见复算脚本和LOCAL_FINAL_ACCEPTANCE.json。','',
'本轮RA平均AUC比Coreset高3.020833pp，但与Random精确持平（79.6875%）；RA跨seed AUC的样本SD为6.853660pp，高于Random的1.082532pp。RA在42/43上提升，44下降，因此不能称为三seed一致改善或稳定优于Random。RA最终旧任务平均成功率在三个seed均高于Coreset，但均低于对应Random；相对Random三seed平均低2.222222pp。',
'seed44保留了部分Task6收益：最终50%对Coreset15%，但Task8最终60%对70%、Task9初学即70%对85%。按AUC公式，Task6贡献+4.0625pp、Task8贡献−1.25pp、Task9贡献−3.75pp，合计−0.9375pp。这说明本轮并非所有旧任务都受益，也未解决Task9新任务表现；它是结果分解，不是机制因果证明。',
'', '## 逐seed配对差与全部矩阵','', '|Seed|对照|ΔAUC|ΔFWT|ΔNBT|ΔOldFinal|ΔFinalAll|','|---:|---|---:|---:|---:|---:|---:|']
for seed,v in a['per_seed'].items():
 for comp in ['coreset','random']:lines.append('|'+seed+'|RA−'+labels[comp]+'|'+'|'.join(f"{v['delta_vs_'+comp][k]['percent']:+.6f}" for k in keys)+'|')
for seed,v in a['per_seed'].items():
 lines+=['',f'### seed{seed} 全部有效单元（每格20条）','','|stage|task|Random|Coreset|RA|','|---:|---:|---:|---:|---:|']
 for x in v['matrix']:lines.append('|'+ '|'.join(str(x[k]) for k in ['stage','task','random','coreset','ra_coreset'])+'|')
 negative=[f"stage{x['stage']}/Task{x['task']} {x['ra_coreset']-x['coreset']:+g}pp" for x in v['matrix'] if x['ra_coreset']<x['coreset']]
 lines+=['','RA低于Coreset的单元：'+('；'.join(negative) if negative else '无')+'。']
lines+=['','## 入库与计算成本','','Task6首次入库前，三个seed的RA实际模型tensor及初始化全RNG均与对应父Coreset一致。所有任务实际有序入库ID与冻结offline一致，固定配额、半配额Random锚点、最少交换数及载荷哈希链全部通过。IFA仍按实际新记忆运行，公式不变不等于有效输入不变；逐seed实际IFA审计原文保留在details.ifa及各AUDIT.json。','', '|Seed|Task6/7/8/9交换数|训练分钟|评估分钟|验收分钟|','|---:|---|---:|---:|---:|']
for seed,v in a['per_seed'].items():
 tm={x['phase']:x['seconds']/60 for x in v['timings']}
 lines.append(f"|{seed}|"+'/'.join(str(v['storage'][str(t)]['swaps']) for t in range(6,10))+f"|{tm['train']:.3f}|{tm['evaluate']:.3f}|{tm['audit']:.3f}|")
lines+=['','这些是正式阶段墙钟耗时，不包含下载、历史基线成本和排队；不声称完整历史GPU计费。每任务的原/新最大覆盖半径、平均最近距离及组成诊断保留在details.selection中；RA不保证几何半径不增。Task9入库发生在其训练之后，不是本轮Task9成功率的直接训练原因。','', '## 真实执行与只读复算','', '远端队列自动运行各seed admission → supervise(train/evaluate/audit/exit acceptance)，随后aggregate_three_seeds.py及final_queue_audit.py；命令、时间、退出码在queue/commands.jsonl、各seed/evidence/commands.jsonl和timings.jsonl。预检仅既有每seed4次隔离optimizer更新、0环境，没有为本次验收追加测试或rollout。','', '本地执行：', '```text', 'python check_progress.py --package', f'python prepare_final_evidence.py {D.name}', f'python audit_progress.py {D.name} --all-seeds', f'python final_local_audit.py {D.name}', f'python write_final_summary.py {D.name}', '```', '上述成功退出0。轻量包按PACKAGE_MANIFEST逐文件核对，baseline记录绑定冻结合同输入hash，逐episode对vector账目、初态、checkpoint身份复核，三seed统计与服务器aggregate精确一致。若后续退出快照晚于package采集，原SNAPSHOT保留，FINAL_SNAPSHOT/queue_final单独保存。','', '## 解释边界','', '这是看过seed42结果后授权的43/44探索性扩展，仅三个训练seed，不能据此声称统计显著或跨套件普遍有效。不同episode的动作采样仍共享stage全局RNG，初态匹配不代表每条轨迹噪声隔离。入库成员变化会经原路径排序改变实际batch，也可能间接改变IFA；不能把所有差异归为纯几何覆盖收益。官方预训练state模态元数据未知（DISCLOSED_GAP），本轮有效state=True且参数完整加载；warmup/调度/最终ep99规则含公开信息之外的工程补全，不能宣称精确恢复作者原始训练轨迹。实验RTX3090与论文/预训练硬件A100的差异保留披露。所有负向结果均列出，不据此改seed、门槛、checkpoint或追加实验。','', '完成本轮交付后删除定时检查，不自动启动下一轮。仅发布代码、报告、身份/哈希元数据及episode结果，不发布权重、示范、latent、buffer载荷或凭据。']
(D/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
status=dict(status='COMPLETED',local_acceptance='PASS',seeds=[42,43,44],epochs=1200,steps=228300,checkpoints=12,formal_records=600,vector_attempts=60,queue_exit=0,wrapper_exit=0,report=str(D/'FINAL_REPORT.md'),verified_at=datetime.datetime.now().astimezone().isoformat())
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8');print(json.dumps(status,indent=2))
