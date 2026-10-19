"""Render final read-only acceptance into a reproducible report."""
from pathlib import Path
import json,datetime
R=Path(__file__).resolve().parent
a=json.loads((R/'LOCAL_FINAL_ACCEPTANCE.json').read_text());assert a['status']=='PASS'
now=datetime.datetime.now().astimezone().isoformat()
rows=['# RA-Coreset GOAL seed42 最终报告','',f'本地验收时间：{now}。状态：COMPLETED；三个指标门槛全部通过。按冻结协议 no_auto_extend，当前未启动其他 seed 或 OBJECT。','',
'400/400 epoch、76100/76100 optimizer step、4个 ep99 checkpoint 已在服务器严格加载100个下游键；200/200正式episode、20/20 vector attempts、0技术异常。训练、评估、audit、pipeline、supervisor退出码全部0，服务器退出验收和本地独立复算均PASS。重新连接后的真实进程检查无残留，冻结source/config/input/feature/modal哈希PASS。未追加训练或评估。','',
'## 指标（百分数；NBT沿用原有符号定义）','',
'| 方法 | AUC | FWT | NBT | OldFinal | FinalAll |','|---|---:|---:|---:|---:|---:|']
for n in ['Random','Coreset','RA-Coreset']:
 rows.append('| '+n+' | '+' | '.join(f"{a['metrics'][n][k]['percent']:.6f}" for k in ['AUC','FWT','NBT','OldFinal','FinalAll'])+' |')
rows+=['','RA相对Coreset：AUC +3.75pp，FWT +3.75pp，NBT不变，OldFinal +6.666667pp，FinalAll +6.25pp。相对Random：AUC +1.5625pp，但OldFinal低1.666667pp，FinalAll相同。精确分数及差值见LOCAL_FINAL_ACCEPTANCE.json。','',
'预定门槛：AUC=33/40 > 63/80；FWT=17/20 >= 13/16；OldFinal=3/4 >= 41/60；工程PASS。均通过。','',
'## 全部闭环单元（成功率%，每格20条）','','| 已学阶段 | 评估任务 | Random | Coreset | RA-Coreset |','|---:|---:|---:|---:|---:|']
for x in a['matrix']:
 rows.append(f"| {x['stage']} | {x['task']} | {x['Random']:g} | {x['Coreset']:g} | {x['RA-Coreset']:g} |")
rows+=['','相对Coreset的AUC增量，Task6/7/8/9分别贡献0/0/+2.5/+1.25pp。Task6在stage7由65%降至50%，stage8升至70%，最终由45%升至55%；这些变化抵消后，其四阶段平均仍为60%。不能将整体改善写成Task6遗忘已经解决：Task6最终仍低于Random的70%。Task8刚学完及最终均提高10pp，Task9由90%升至95%（多1次成功）。','',
'## 真实入库与输入保护','','| Task | 配额 | 随机锚点 | 最少交换 | 原最大距离 | RA最大距离 | 原平均距离 | RA平均距离 |','|---:|---:|---:|---:|---:|---:|---:|---:|']
for t,x in a['storage'].items():
 rows.append(f"| {t} | {x['quota']} | {x['anchor_count']} | {x['swap_count']} | {x['max_before']:.8f} | {x['max_after']:.8f} | {x['mean_before']:.8f} | {x['mean_after']:.8f} |")
rows+=['','四任务实际有序成员与冻结离线名单完全一致，锚点全部保留、交换次数为缺失锚点的最少数，2381个累积buffer文件的元数据哈希链一致；服务器已检查原载荷哈希。Task6首次入库前实际模型张量、初始化RNG与父Coreset一致。全部任务IFA选中任务对相同，但实际记忆/参考特征和训练轨迹可能变化，不能据此声称IFA输入完全相同。Task9自己的入库发生在其训练后，不影响本次已完成模型。','',
'覆盖不是硬约束：RA四任务最大距离均比原Coreset增大。本轮结果不证明几何覆盖指标越好，闭环就必然越好。','',
'## 时间、命令与验收证据','',
'训练约40.56分钟，评估约68.73分钟，服务器audit约0.58分钟。详细计时和真实阶段命令见final_evidence/timings.jsonl、commands.jsonl。',
'',
'本次读取命令：`python execution/20261018_ra_coreset_goal_seed42/check_status.py`（最终退出0）；`python execution/20261018_ra_coreset_goal_seed42/collect_completion.py`（退出0）。其远端命令及退出码见COLLECTION_RECEIPT.json。只运行hash校验、monitor_snapshot.py和package_light.py，没有调用训练/评估入口。',
'',
'本地复算：`python execution/20261018_ra_coreset_goal_seed42/prepare_comparison_evidence.py`；`python execution/20261018_ra_coreset_goal_seed42/recompute_final.py`，均退出0。下载91份轻量包内文件，逐个匹配PACKAGE_MANIFEST；父Random/Coreset证据逐文件匹配冻结合同。四个checkpoint严格加载依据服务器AUDIT，不下载权重。',
'',
'首次本地复算因运行时storage不含offline-only diagnostics字段退出1；已记录LOCAL_RECOMPUTE_REPAIR.json，改为从哈希已验证的offline清单读取描述性诊断后退出0。此修复仅在本地报告代码，无训练、评估或冻结服务器源码改动。',
'',
'服务器刚启动时发生SSH banner EOF/连接关闭；先前TCP探测保存在monitor_20261019_1200。随后重连成功，不将连接失败误记为实验失败。原部署阶段两次安装器失败证据保留在launch历史中。自动化ra-coreset-goal-seed42已在完整验收后删除。',
'',
'## 解释边界','',
'这是多轮探索后的一次seed42结果，不能据此声称统计显著或跨seed稳定。每格20条，单次成功对应5pp。原动作采样仍为每stage重置的全局RNG，未隔离逐轨迹随机流。成员改变会经原路径排序改变实际batch序列；IFA对相同也不隔离参考内容变化，无法把收益纯粹归因于随机锚点的某一个机制。官方预训练state模态元数据unknown，启用state=True依论文描述；调度/warmup和ep99规则为既有协议补全。Random/Coreset旧基线只读复用，未重训或重评。','',
'发布仅代码、报告、哈希/ID和episode结果等轻量证据；不包含checkpoint、RNG二进制、latent、buffer载荷、演示、原专家动作或凭据。']
(R/'FINAL_REPORT.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
(R/'RUN_STATUS.json').write_text(json.dumps(dict(status='COMPLETED',observation_time=now,engineering='PASS',metric_gate=True,decision=a['decision'],epochs=400,optimizer_steps=76100,formal_records=200,vector_attempts=20,auto_extend=False),indent=2)+'\n',encoding='utf-8')
print('FINAL_REPORT written')
