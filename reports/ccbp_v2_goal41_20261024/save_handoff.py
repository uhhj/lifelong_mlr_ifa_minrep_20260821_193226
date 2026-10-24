"""Save actual dispatch/progress; never deploy or change frozen scientific code."""
import json,datetime,hashlib,ast
from pathlib import Path
R=Path(__file__).resolve().parent
dp=R/'deploy_remote_20261024_233415/EVIDENCE.json';sp=R/'check_readonly_20261024_233606';dep=json.loads(dp.read_text());obs=json.loads((sp/'EVIDENCE.json').read_text());audit=json.loads((sp/'LOCAL_AUDIT.json').read_text());adm=json.loads((sp/'ADMISSION_AND_WEIGHT_AUDIT.json').read_text())
assert audit['status']=='PASS_PROGRESS_NOT_FINAL' and dep['contract_sha256']==audit['contract']
active={k:dep[k] for k in ['root','private_root','private_data','controller','contract_sha256','frozen_files']};active['automation_id']=None
(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,indent=2))
status=dict(status='PARTIAL_TRAINING',observation_unix=obs['observed'],observation_cst=datetime.datetime.fromtimestamp(obs['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),phase=audit['phase'],counts=audit['counts'],contract=audit['contract'],latest_evidence=sp.name,own_admission=adm,formal_updates=sum(c['steps'] for c in audit['counts'].values()),formal_episodes=0,budget=dict(arms=list('BCNV'),epochs=400,main_updates=134800,formal_episodes=320),capacity_observed=obs['capacity'],next_check_minutes=100,eta='Approximately90–105min remaining whole training from23:36; later320episodes roughly40min by old comparable throughput, Task6/7 and N/V unmeasured. Final audit extra.',automatic_expansion=False)
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2))
header='''# CCBP-v2 GOAL seed41：四组有限验证

## 2026-10-24 23:36:18 CST 实际启动与准入（优先）

科学root=/data/mlr_ifa/experiment_workspaces/CCBP_V2_GOAL41_20261024_v1，controller19552、GPU19723健康，32冻结项及父源一致，合同fa6886806120e9cff7547fc0723b602e4fd53497c0a9c1d433618e50be267e47。当前PARTIAL_TRAINING，B正式3完整epoch1011更新；C/N/V尚无正式更新，正式环境0。不重跑deploy/controller。

四组自己的GPU准入全部实际PASS：B/C各5、N/V各4，共18隔离optimizer调用/0环境；完整下一步序列化恢复精确，B/C相对旧辅助实现完整更新精确。固定分项梯度测量开关前后完整更新精确；准入额外16次autograd测量单列。新存储别名和序列化实际346294013bytes小于350MB界，恢复与final state_dict共享tensor storage，未缩减恢复字段。原型自己的CPU检查及各组实际合成检查也通过，未套用原型自述PASS。

教师旧430合法窗口原地只读复用，当前额外教师模型前向0；历史14前向/0.914秒只作继承成本，不能重复记成本轮实际前向。tau=0.0038629346527159214，按全部430窗口一次冻结；w最小0.000871729、最大0.829706132、均值0.478685170。权重JSON SHA c1bcd3cf8d3f20a5fd96028e588ce6c9c3ef8d617308593a0b844de64b0e4357；本地中位数与全部权重标量独立复算绑定通过，不能当闭环收益。

第一批B三个epoch平均13.505554秒。四组剩余397轮按该速度线性89.36分钟，恢复保存/准备另计，C/N/V本轮吞吐未测；整段训练暂按90–105分钟，下一100分钟后约10月25日01:15–01:20检查，不内部阶段短轮询。320回合评估仅据旧Task8/9估约40分钟量级，新增Task6/7未测，最终审计额外，不承诺总完成时间。

当前/root普通1888186368B、/data root3457196032B，已写B首个恢复态；仍保留1/2GiB。部署前每volume1.04GiB额外峰值界已留足，按唯一inode抵扣，硬链接不重复算信用，无删除/降预算/付费。

协议在正式更新前发布核远端18193d9ce99a9fbd1fc19abc5a1a5e6d08789827。初次预发布在push后本地receipt因尚无RUN_STATUS而失败，事实保留，不冒充成功回执；修正本地源换行SHA比较与发布前状态读取后才取得上述完整回执。首次Python3.8原型类型别名导入失败也保留，随后仅类型注解兼容修复、公式与梯度一致。新正式部署一次成功，无科学重试。

新的final_readonly/audit_final_local只是AST检查，尚待全队列自然exit0后实际4CPU strict/完整恢复硬链接/旧B/C最终tensor对照/400epoch134800步/320episode32vector16cell/初态真实字节/噪声/weights/J/route及梯度成本验收。未完成不称最终PASS。

以下为部署前登记记录，不代表当前阶段：

'''
old=(R/'REPORT.md').read_text(encoding='utf8');(R/'REPORT.md').write_text(header+old,encoding='utf8')
handoff='''继续已授权CCBP-v2 GOAL seed41四组有限验证，先读本目录USER_METHOD/PROTOCOL/IMPLEMENTATION_NOTES/REPORT/RUN_STATUS/ACTIVE_ROOT及最新证据。唯一科学root /data/mlr_ifa/experiment_workspaces/CCBP_V2_GOAL41_20261024_v1，controller19552，合同fa6886806120e9cff7547fc0723b602e4fd53497c0a9c1d433618e50be267e47，32冻结项。禁止重跑deploy/build/controller或已完成科学阶段。旧v1/ALIGN/CAET/TACR/readout/语义筛查不恢复。
最新23:36:18 check_readonly_20261024_233606 LOCAL_AUDIT及ADMISSION_AND_WEIGHT_AUDIT真实PASS_PROGRESS_NOT_FINAL，四组自己GPU18隔离更新与完整恢复/测量开关精确、B/C旧实现更新精确通过。B3完整epoch1011主步、其余正式0，GPU19723健康，无环境。teacher430 cache原地只读复用，本次额外教师前向0；tau0.0038629346527159214与所有w本地独立复算通过。
四组B/C/N/V从共同Stage8重新训练Task9各100epoch33700步，全134800。B固定蒸馏、C旧专家cost路由、N关闭教师锚的新路由、V完整v2；lambda.1/beta.1/rho1、原mask/RA人口3365/3218有效147屏蔽430旧Task8、FP32/b10/IFA/模型不改。全部训练后Task6–9各20初态，共320回合32vector16cell，配对噪声原argmax Normal600步5zero。V主要Task8>C且Task9>=C，Task6/7副作用全报，不以mean掩盖；不加uniform，不自动扩seed、不扫描。
用户定时按预计整段训练或整段评估完成点，不固定短周期。当前B3轮13.505554秒均值，其余组未测，剩训练90–105分钟参考；下一100分钟。320评估旧可比约40min但Task6/7未测，最终审计另计。每次醒来remote.py check_readonly.py→audit_progress.py SNAPSHOT；评价阶段再audit_evaluation_progress.py。保持健康不重启，一轮保存发布后结束不守候。
新checkpoint/recovery同一文件storage别名，final硬链接只在100轮后创建；B/N在/root，C/V在/data，layout为准。每volume1.04GiB峰值界，唯一inode已分配抵扣；system普通1GiB/data root2GiB保留。不删旧资产、不下载payload、不付费或减预算。
自然controller.exit0且无自身进程后remote.py final_readonly.py→audit_final_local.py FINAL；新的最终路径仅AST尚未实际PASS，外置错误保留修复，不重训重评。验收4CPU100键/恢复字段/inode/旧B/C最终tensor/所有初态文件字节与noiseSHA/weights中位数/J/route/400epoch134800步/320回合及全部分项梯度成本，报告V对B/C/N的Task6–9、SF/FS和代理一致性。固定epoch0/9/49/99梯度测量每组4batch、正式额外64autograd，准入额外16另列，不称等算力。完成真实验收报告发布后删除本监控，无论科学正负均不自动扩展。
SSH root@jq1.9gpu.com:11920已授权，口令仅现有内存/getpass不文件日志参数。故障先核停止/hash/完整epoch状态留证后版本化工程修复，不盲重启正式评估，不覆盖冻结源。初CPU类型注解失败及本地发布receipt错误均已留证，科学训练未重试。
每次检查/验收按AGENTS经本目录publish.py精确审查轻量type/size/secret/hash，串行stage非force推MLR并核remoteSHA/localreceipt。TLS可OpenSSL+HTTP1.1仍验证证书；禁模型/CP/latent/rawactions/states/demos/buffer/RNG/privatepayload/未审查zip。健康无变化安静，仅实质变化/失败/完成/需行动通知。
'''
(R/'MONITOR_HANDOFF.md').write_text(handoff,encoding='utf8')
for p in R.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
(R/'STATIC_REVIEW.json').write_text(json.dumps(dict(status='AST_PASS_OWN_GPU_ADMISSION_ACTUAL_FINAL_PATH_PENDING',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')},admission_audit_sha256=hashlib.sha256((sp/'ADMISSION_AND_WEIGHT_AUDIT.json').read_bytes()).hexdigest()),indent=2))
print(json.dumps(status,indent=2))
