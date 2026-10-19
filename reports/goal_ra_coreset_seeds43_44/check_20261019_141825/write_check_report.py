"""Write a new timestamped progress report without replacing old reports."""
from pathlib import Path
import json,sys,datetime
R=Path(__file__).resolve().parent;D=R/sys.argv[1]
a=json.loads((D/'PROGRESS_ACCEPTANCE.json').read_text());snap=json.loads((D/'SNAPSHOT.json').read_text())
stamp=datetime.datetime.fromtimestamp(snap['time'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
c=a['completed']['43'];p=a['partial']['44'];eta=a['eta']
text=['# RA-Coreset GOAL 43/44：进度与seed43独立复算','',f'快照时间：{stamp}。整体仍为PARTIAL；本次只读核查，没有启动训练、追加评估或重复预检。各package采集稍晚于快照，训练计数以下注明来源。','',
'seed43已完成400epoch、76100step、4个严格加载的ep99模型、200条正式记录及20个完整vector attempts。训练/evaluate/audit/pipeline/supervisor均退出0，服务器PROCESS_EXIT_ACCEPTANCE及本地独立复算PASS；0技术异常。seed44已自动启动，准入与4次隔离GPU更新恢复PASS，0环境预检；正式评估尚未开始。','',
f"seed44包内epoch证据为{p['epochs']}/400、{p['steps']}/76100step；初始快照为{snap['seeds']['44']['epochs']}epoch。Task6尚未完成时，最终模型tensor对齐检查为PENDING，不是失败；初始化和最终tensor将由冻结训练入口强制核对。当前无队列失败、GPU正常工作。",'',
'## seed43指标（百分数，精确值见PROGRESS_ACCEPTANCE.json）','','| 方法 | AUC | FWT | NBT | OldFinal | FinalAll |','|---|---:|---:|---:|---:|---:|']
for arm in ['random','coreset','ra_coreset']:text.append('| '+arm+' | '+' | '.join(f"{c['metrics'][arm][k]['percent']:.6f}" for k in ['AUC','FWT','NBT','OldFinal','FinalAll'])+' |')
text+=['','RA相对Coreset：AUC +6.25pp，FWT不变，NBT降低15pp，OldFinal +11.666667pp，FinalAll +10pp。相对Random：AUC +5.625pp，但OldFinal低3.333333pp。不得将这些结果写成每项指标均优于Random。','',
'## seed43全部有效矩阵单元（每格20条）','','| stage | task | Random | Coreset | RA |','|---:|---:|---:|---:|---:|']
for x in c['matrix']:text.append(f"| {x['stage']} | {x['task']} | {x['random']:g} | {x['coreset']:g} | {x['ra_coreset']:g} |")
text+=['','Task8刚学完RA65%，低于Coreset70%；最终则80%高于50%。Task6最终75%高于Coreset70%，但低于Random95%。这些负向或未超越结果全部保留。Task7各已见阶段保持100%，Task9 RA90%/Coreset85%/Random70%。','',
'## 只读一致性与时间估计','',
'seed43实际有序入库与冻结offline完全一致，四任务交换121/106/100/249，配额499/445/452/985，锚点全部满足，累积2381载荷哈希链一致；Task6实际模型tensor和初始化全RNG同父Coreset。两seed冻结源码/config/modal/input/feature/orchestration校验PASS，全部下载文件通过PACKAGE_MANIFEST。父结果逐文件对照新合同哈希；逐episode、vector账目、CP身份、初态身份及精确指标独立复算PASS。','',
f"seed43实际训练{c['timings'][0]['seconds']/60:.2f}分钟，评估{c['timings'][1]['seconds']/60:.2f}分钟。按44最近20个epoch与43同位置实测时间比{eta['training_recent_rate_ratio']:.4f}，估计44剩余训练约{eta['training_remaining_minutes']:.1f}分钟；评估参考约{eta['evaluation_reference_minutes']:.1f}分钟。含验收余量，快照时剩余约{eta['total_remaining_range_minutes'][0]}–{eta['total_remaining_range_minutes'][1]}分钟，预计北京时间15:50–16:25完成。可能随真实吞吐变化，未提前最终验收。",'',
'本次真实读取命令为check_progress.py --package（exit0）；其远端monitor_snapshot、verify及package_light命令/exit0详见COMMANDS.json。本地audit_progress.py progress_20261019_141825（exit0），仅复算元数据与结果；没有模型加载或环境调用。保持两小时检查，不持续守候、不重复启动。','',
'等待44完整结束、queue/wrapper退出0及三seed最终验收后再汇总三seed均值/样本SD。当前不可根据两个已完成seed作三seed结论。该扩展在看到42后授权；global动作RNG未隔离、batch排序、IFA间接变化、pretrain state unknown及调度补全边界不变。']
(D/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
status=dict(status='PARTIAL_SEED43_COMPLETED_SEED44_TRAINING',observation_time=stamp,seeds={'43':dict(status='COMPLETED_AUDIT_PASS',epochs=400,steps=76100,records=200),'44':dict(status='TRAINING',epochs=p['epochs'],steps=p['steps'],records=0,epoch_count_source='metadata package')},eta=eta,automation='ra-coreset-goal-seed43-44-seed',latest_evidence=str(D),queue_exit=snap['queue_exit'],wrapper_exit=snap['wrapper_exit'])
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
print(json.dumps(status,indent=2))
