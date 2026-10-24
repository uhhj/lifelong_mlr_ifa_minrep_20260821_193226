from pathlib import Path
import ast,json,hashlib,datetime
R=Path(__file__).resolve().parent
files=['runtime.py','layout.py','legacy_losses.py','prototype_loss.py','losses.py','synthetic_checks.py','train.py','evaluate.py','controller.py','PROTOCOL.md','USER_METHOD.txt']
bundle={n:(R/n).read_text(encoding='utf8') for n in files}
for n,s in bundle.items():
 if n.endswith('.py'):ast.parse(s)
(R/'DEPLOY_TEXT_BUNDLE.json').write_text(json.dumps(bundle,ensure_ascii=True),encoding='utf8')
cpu=sorted(R.glob('cpu_admission_readonly_*/EVIDENCE.json'))[-1];v=json.loads(cpu.read_text());assert v['status']=='PASS_LOCAL_SOURCE_CPU_ADMISSION'
for n,h in v['source_sha256'].items():assert hashlib.sha256((R/(n+'.py')).read_text(encoding='utf8').encode()).hexdigest()==h
status=dict(status='PREREGISTERED_CPU_ADMITTED_GPU_PENDING',observation_unix=v['observed'],root='/data/mlr_ifa/experiment_workspaces/CCBP_V2_GOAL41_20261024_v1',controller=None,budget=dict(arms=list('BCNV'),epochs=400,main_updates=134800,formal_episodes=320),cpu_admission=str(cpu.parent.name),gpu_admission=False,formal_updates=0,formal_episodes=0,automatic_expansion=False)
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf8')
(R/'REPORT.md').write_text('''# CCBP-v2 GOAL seed41：四组有限验证

当前为PREREGISTERED_CPU_ADMITTED_GPU_PENDING，未启动正式训练或环境。用户授权的B/C/N/V从共同Stage8独立训练Task9，各100epoch33700主更新，之后评估Task6–9共320正式回合；不扩seed、不扫参数。PROTOCOL完整固定主要比较、新任务零允许损失及其他旧任务副作用报告。

已实际读取用户ZIP并逐文件审查，SHA见PROTOTYPE_REVIEW。服务器CPU合成准入实际通过：N/V损失与梯度和上传原型一致，B/C与旧版一致，teacher无梯度、连续纠错、严重度路由、并列与重编号通过。上传自述PASS没有替代本轮准入。首CPU探针因原型Python3.9运行时类型别名不兼容服务器3.8，在导入阶段失败；仅改typing.Tuple后通过，0正式训练/环境，原失败保留。

GPU与四组自己的完整状态恢复尚待执行，不能套用旧PASS。计划18隔离optimizer更新（B/C各5、N/V各4），正式134800另计；每组固定4个batch的分项梯度测量额外autograd计算单列，不能称等算力。

只读预检已核旧队列自然exit0、无残留/GPU空闲、父合同和Stage8初始化/输入不变。当前/root普通空闲2234621952B、/data root空闲3457536000B。新输出分两volume，各1.04GiB峰值界含原子保存，保留1/2GiB。完整恢复与finalCP共享storage/最终inode，全部状态保留；自己的GPU准入须实际核序列化大小与精确恢复。不删旧资产、不下载权重/payload、不新增付费。

训练先用上一轮B/C各约21分钟作为量级参考，四组约85–100分钟，N/V吞吐尚未测；准备/保存另计。随后320回合评估用旧Task8/9总18.65分钟翻倍仅为约40分钟量级参考，Task6/7吞吐未测；最终独立验收额外。监控按整段预计结束点设置，不固定短周期。完成全部实际验收与发布后停止。
''',encoding='utf8')
(R/'STATIC_REVIEW.json').write_text(json.dumps(dict(status='AST_AND_CPU_PASS_GPU_PENDING',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')},cpu_evidence_sha256=hashlib.sha256(cpu.read_bytes()).hexdigest()),indent=2),encoding='utf8')
print(json.dumps(status,indent=2))
