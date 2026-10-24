"""Write truthful observation and single-monitor handoff from a completed local audit."""
from pathlib import Path
import sys,json,datetime
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);a=json.loads((D/'LOCAL_AUDIT.json').read_text(encoding='utf8'));v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'))
root='/data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1'
if (R/'ACTIVE_ROOT.json').exists():root=json.loads((R/'ACTIVE_ROOT.json').read_text())['root']
ts=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
status=dict(status='PARTIAL_TRAINING' if not a['failure'] else 'PARTIAL_ENGINEERING_FAILURE',observation=ts,observation_unix=a['observation'],root=root,contract=a['contract'],counts=a['counts'],own_admissions=a['admissions'],isolated_optimizer_calls=a['isolated_optimizer_calls'],teacher_bindings=a['teacher_bindings'],formal_episodes=a['formal_episodes'],phase=a['phase'],latest_evidence=D.name,capacity=v.get('capacity'),budget=dict(epochs=1600,steps=304400,episodes=800),automatic_expansion=False,next_check_minutes=55,final_acceptance='PENDING')
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf8')
if not (R/'ACTIVE_ROOT.json').exists():(R/'ACTIVE_ROOT.json').write_text(json.dumps(dict(root=root,controller=20181,contract=a['contract'],automation_id=None),indent=2),encoding='utf8')
summary='# 最新实际进度 '+ts+'\n\n'+json.dumps({k:status[k] for k in ['status','counts','own_admissions','formal_episodes','latest_evidence']},ensure_ascii=False,indent=2)+'\n\n只代表完整epoch账目和自身准入，不代表闭环收益或整体接受。\n\n'
(R/'REPORT.md').write_text(summary+(R/'REPORT.md').read_text(encoding='utf8'),encoding='utf8')
prompt='''继续用户授权CCBP-v2完整GOAL seed41四组，用户离开并明确要求自主解决工程障碍直至全部验收发布完成。先完整读取D:/论文课题研究/MLR+IFA/execution/20261024_ccbp_v2_full的PROTOCOL、IMPLEMENTATION_NOTES、REPORT、RUN_STATUS、ACTIVE_ROOT及最新证据。实际root/controller/合同以ACTIVE_ROOT和最新证据为准，禁止重跑deploy/build/controller或重训已完成阶段。旧Task9分叉已废止exit1/9完整epoch，原监控已删除；旧ALIGN/CAET/TACR/readout不恢复。
四组B/C/N/V各自官方GOAL初始化，完整Task6→7→8→9各100epoch，单组76100主步、全1600epoch304400主步。原GOAL人口997/1388/1848/3365，合法917/1286/1723/3218；合法旧窗口0/477/899/1329，FP32/b10/sampler0，原RA成员/ALIGNmask/BC+.1IFA及模型不改。Task6辅助0；Task7–9各自上一阶段教师eval/no_grad/cache冻结，全部合法旧窗口，尺度旧训练动作RMS floor.1，tau教师cost中位数冻结。B原教师蒸馏、C原专家cost，N新路由无锚，V一致度软锚。lambda.1/beta.1/rho1不扫参。自己的每阶段隔离准入、完整恢复和状态化梯度测量开关精确，不能套旧PASS。
当前自己的实测进度见RUN_STATUS。首次B Task6约2.92秒/轮；后续辅助阶段未测，借此前Task9约13.35秒/337batch作规划，B整组剩余约45–55min含准备保存。下一55分钟在整组训练结束附近检查，不内部Task切换短轮询；以后依实际整组训练/整段评估剩余时间调同一监控。健康无变化安静，一轮只读检查/审计/保存/发布后结束不守候。
remote.py check_readonly.py → audit_progress.py SNAPSHOT。若正常，按整组剩余ETA更新监控；若故障先核进程停/合同一致/完整epoch恢复态/失败保留，再新版本工程修复，不覆盖冻结科学源码，不盲重启正式评估。原分叉因检查p.grad必须None误判已有零tensor，完整版已修精确前后比较并在有历史更新状态下准入，不改目标。
全四组训练后800正式episode80vector40cell：各阶段评所有已见任务，每cell20初态0–19，同task/index配对Normal噪声，argmax分量600步5zero，无测试重试。完整controller.exit0无自身进程后remote.py final_readonly.py→audit_final_local.py FINAL。新的外置路径只AST，必须真实16CPU严格100键/完整恢复/history/Task6官方ALIGN和四组tensor相等/teacher缓存与本组前阶段CP/原RA成员标签/800初态真实字节与noiseSHA/成本标量独立复算。外置wrapper错误留证修，不重训重评。完成后给出V-C/V-N/V-B全矩阵、AUC/FWT/NBT低好/OldFinal/FinalAll、负单元与SF/FS、离线代理一致性及限制。主要探索门仍V最终Task8>C且Task9>=C，Task6/7副作用单报；不自动扩seed或任何权重扫描，单seed重复探索不称显著/硬保护。
用户授权自动删除废用数据，已精确删除三份旧中间恢复释放约1.079GB，receipt和旧最终资产留存绑定在AUTHORIZED_CLEANUP_RECEIPT。旧最终模型/最终恢复/原始数据不能当废用自动删。历史中断恢复演练已退休，不伪报原中间payload仍在。新存储layout为准，root1.07GiB/data1.88GiB新增峰值、唯一inode抵扣，system普通留1GiB/data root留2GiB。不得降预算精度、下载weights/features/rawactions/states、新增付费。若新容量障碍可先只读查明确废用中间资产并精确留证清理，不碰未验收/活跃/最终资产。
SSH root@jq1.9gpu.com:11920已授权，凭据仅现有内存/getpass，不文件日志参数。每轮本目录publish.py按AGENTS精确审查轻量type/size/secret/hash串行stage非force推MLR并核remoteSHA/localreceipt，TLS OpenSSL/HTTP1.1仍验证证书。禁模型/CP/原demo/action/state/latent/buffer/RNG/privatepayload/未审查bundle。发布进度不等于验收。全部独立验收报告和机制解释发布后删除本监控；只实质变化、失败、完成、需行动通知。
'''
(R/'MONITOR_HANDOFF.md').write_text(prompt+'\n最新观察：'+summary,encoding='utf8');print(ts)
