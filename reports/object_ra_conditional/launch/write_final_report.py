"""Render the verified independent result, without changing scientific inputs."""
from pathlib import Path
import json,datetime
R=Path(__file__).resolve().parent;C=R/'completion_20261020';D=C/'download'
j=lambda p:json.loads(p.read_text(encoding='utf-8'))
a=j(C/'LOCAL_FINAL_ACCEPTANCE.json');assert a['status']=='PASS'
keys=['AUC','FWT','NBT','OldFinal','FinalAll']
names={'random':'Random＋IFA','coreset':'Coreset＋IFA','ra_coreset':'RA-Coreset＋IFA','ra_coreset_ifa_zrv2':'RA-Coreset＋ZR-v2＋IFA'}
tz=datetime.timezone(datetime.timedelta(hours=8))
end=datetime.datetime.fromtimestamp(j(D/'current/FINAL_SERVER_ACCEPTANCE.json')['time'],tz)
now=datetime.datetime.now(tz)
lines=['# OBJECT RA-Coreset → ZR-v2 条件实验最终报告','',
       '**COMPLETED：工程验收通过，第二关性能门槛未通过，按既定条件停止。**',
       '',f'服务器完成时间：{end:%Y-%m-%d %H:%M:%S} CST；本地独立验收：{now:%Y-%m-%d %H:%M:%S} CST。',
       '',
       'RA＋IFA 的41/42/43三seed均优于匹配Random＋IFA，平均AUC增加4.93056个百分点，第一关通过。自动追加的RA＋ZR-v2＋IFA只有seed43提高，平均AUC下降1.18056个百分点，第二关未通过。未启动GOAL RA组合，也未启动仅在第一关失败时才授权的GOAL Coreset备用组合。负结果不影响工程完成验收。',
       '',
       '## 三seed AUC（%）','',
       '|Seed|Random＋IFA|Coreset＋IFA|RA＋IFA|RA＋ZR-v2＋IFA|RA−Random（pp）|组合−RA（pp）|',
       '|---|---:|---:|---:|---:|---:|---:|']
for i,s in enumerate(['41','42','43']):
 vals=[a['seed_results'][s][arm]['metrics']['AUC']['percent'] for arm in names]
 lines+=['|'+s+'|'+'|'.join(f'{x:.5f}' for x in vals+[a['gates']['stage1']['paired_delta_pp'][i],a['gates']['stage2']['paired_delta_pp'][i]])+'|']
lines+=['|均值 ± 样本SD|'+'|'.join(f"{a['statistics'][arm]['AUC']['mean_percent']:.5f} ± {a['statistics'][arm]['AUC']['sample_SD_percent']:.5f}" for arm in names)+'|4.93056 ± 1.51788|−1.18056 ± 3.12558|',
        '', '所有SD均使用ddof=1。两列差值的SD按同seed配对差计算，不能用两组SD相减。OBJECT43 Random/Coreset均采用第二次完整repeat1；首次高分不替换或删除。',
        '', '## 全部指标','', '|Seed|方法|AUC %|FWT %|NBT pp|OldFinal %|FinalAll %|','|---|---|---:|---:|---:|---:|---:|']
for s,rows in a['seed_results'].items():
 for arm,row in rows.items():lines+=['|'+s+'|'+names[arm]+'|'+'|'.join(f"{row['metrics'][k]['percent']:.5f}" for k in keys)+'|']
lines+=['','|方法|AUC 均值±SD|FWT 均值±SD|NBT 均值±SD|OldFinal 均值±SD|FinalAll 均值±SD|','|---|---:|---:|---:|---:|---:|']
for arm,ms in a['statistics'].items():
 lines+=['|'+names[arm]+'|'+'|'.join(f"{ms[k]['mean_percent']:.5f} ± {ms[k]['sample_SD_percent']:.5f}" for k in keys)+'|']
lines+=['','## 所有任务成功率矩阵（%）','',
        '每行是训练/校准完成的阶段，每列是被测任务；每个有效单元20条正式闭环。—表示协议未评估。',
        '', '|Seed|方法|阶段|Task6|Task7|Task8|Task9|','|---|---|---|---:|---:|---:|---:|']
for s,rows in a['seed_results'].items():
 for arm,row in rows.items():
  for i,values in enumerate(row['matrix']):lines+=['|'+s+'|'+names[arm]+'|'+str(i+6)+'|'+'|'.join('—' if v is None else f'{v:.0f}' for v in values)+'|']
lines+=['','## 预算、工程验收与真实退出','',
        '- 新RA：三seed共1200完整epoch、316500逻辑optimizer step、12个ep99最终checkpoint、600条正式记录。',
        '- 新ZR-v2：三seed共120校准epoch、31650 step、12个calibration ep9 checkpoint、600条正式记录；没有重训公共网络或父基线。',
        '- 合计348150逻辑step、1200新增正式闭环、120个完成vector attempt、24份最终checkpoint严格加载证据。额外回放157个未提交步骤单独计入，正式训练物理更新348307次；隔离预检单独记账，不混入正式步数。',
        '- 本地独立解析2400条真实记录：1200新增＋1200复用Random/Coreset历史记录；逐单元身份、计数、分数指标和两道门槛均与服务器一致。',
        '- RA三seed各4次、ZR-v2三seed各4次隔离优化器预检，合计24次，0环境预检。新增正式环境attempt账目没有技术异常或额外正常失败重试。',
        '- RA入库有序ID与离线结果逐项一致，配额与Random硬锚点通过；ZR-v2使用的父checkpoint、初态哈希、模态和每阶段buffer载荷哈希一致。',
        '- ZR-v2三组各16项结构检查、真实语言门准入、冻结公共网络/固定T0与4最终模型加载均通过。没有为了通过而改门限或硬归零。',
        '- 当前controller/wrapper/supervisor/final_audit/package_light均真实退出0；无残留科学worker，GPU空闲。',
        '', '原seed41 Task6末端跨硬件精确相等失败（exit1）和后续Python3.8只读审计兼容失败（exit1）原样保留。按用户授权去除过严停止条件并由完整状态恢复；后一个失败只重跑只读审计（exit0），没有重跑环境。初始模型/RNG与协议检查仍强制。不能将原失败退出码改写为0。',
        '',
        '原轻量打包器为RA与ZR-v2复用了OBJECT_RA_seed{s}前缀，顶层元数据在包内重名；科学产物和分阶段episode记录未改。已保留原包，外置只读补采43份RA元数据到独立RA_STAGE1_seed{s}前缀，并核对全部下载哈希；没有修补输入或重跑实验。',
        '', '## 实际检查命令','',
        '服务器（`/data/mlr_ifa/env/bin/python`）：`deployment_snapshot.py`、`collect_completion_readonly.py`、`supplement_ra_metadata.py`均退出0。后两者只读取完成产物并写独立证据包，不调用模型、优化器或环境。',
        '',
        '本地：`python execution/20261019_ra_object_conditional/audit_completion_local.py` 退出0。脚本校验两个经过审查的轻量包、下载文件清单和所有记录，独立重算指标与门槛；不导入torch或环境。`write_final_report.py`仅渲染本验收结果。',
        '', '## 计算成本','',
        '|Seed|RA已提交epoch训练秒数|ZR-v2校准epoch秒数|ZR-v2全部阶段含评估耗时秒数|',
        '|---|---:|---:|---:|']
for s,cost in a['epoch_seconds'].items():
 ledger=[json.loads(l) for l in (D/'ledger'/('OBJECT_RA_seed'+s)/'timings.jsonl').read_text().splitlines()]
 total=sum(x['seconds'] for x in ledger)
 lines+=['|'+s+'|'+f"{cost['ra_committed_epochs']:.2f}|{cost['zr_calibration_epochs']:.2f}|{total:.2f}|"]
lines+=['',
        '训练epoch计时不含数据准入、入库、评估、初次失败和157步重放，因此不能当整个实验墙钟。ZR-v2全阶段耗时来自实际commands/timings，完整延迟、显存、固定T0开销在各组AUDIT/COST_ISOLATION.json。',
        '', '## 结论边界','',
        'RA相对Random的这组三seed结果符合预定门槛，但不等于已经证明跨套件普遍有效；相对Coreset平均AUC仅高0.9375pp，seed43低3.54167pp，最终旧任务平均成功率也不是全面改善。ZR-v2在RA之上未形成稳定增益，因此本轮停止，不自行开启另一组或调参。',
        '',
        '本轮为看过先前结果后指定的探索性条件实验。历史父基线与新实验存在4090/3090等硬件差异；Task6末端参数差异已披露，不能宣称后续差异全部由入库模块导致。原全局动作RNG未按轨迹隔离，IFA可能随新记忆间接变化；预训练state元数据未知、学习率调度等细节由共同执行协议补全。ZR-v2追加校准和固定T0开销，非等算力比较。三seed不能据此直接断言统计显著性或普遍因果。',
        '',
        '全部矩阵、精确Fraction、源文件哈希和本地验收：`completion_20261020/LOCAL_FINAL_ACCEPTANCE.json`。复算脚本：`audit_completion_local.py`。报告及轻量证据按约定推送MLR，权重/latent/buffer载荷/原actions/RNG二进制不发布。完成交付后取消本次定时检查。']
(R/'FINAL_REPORT_20261020.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
status=dict(status='COMPLETED',time=now.isoformat(),server_completed=end.isoformat(),executed=['OBJECT_RA_IFA','OBJECT_RA_ZRv2_IFA'],new_formal_records=1200,strict_checkpoints=24,logical_optimizer_steps=348150,extra_replayed_physical_steps=157,stage1_gate=a['gates']['stage1'],stage2_gate=a['gates']['stage2'],GOAL_started=False,stop_reason='STAGE2_METRIC_GATE_NOT_MET',local_acceptance='completion_20261020/LOCAL_FINAL_ACCEPTANCE.json',final_report='FINAL_REPORT_20261020.md',original_failures_preserved=True,publication='pending',automation='delete_after_publication')
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
print('FINAL_REPORT_20261020.md and RUN_STATUS.json written')
