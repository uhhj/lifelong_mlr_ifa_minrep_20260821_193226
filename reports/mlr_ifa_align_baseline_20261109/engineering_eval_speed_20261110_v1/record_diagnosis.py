"""Recompute the diagnosis from actual observations; no science execution."""
from pathlib import Path
import json,datetime,hashlib,statistics
R=Path(__file__).resolve().parent;B=R.parent
J=lambda p:json.loads(p.read_text(encoding='utf8'))
D=B/'engineering_assets_resume_20261110_v2/check_readonly_20261110_162039'
e=J(D/'EVIDENCE.json');a=J(D/'LOCAL_AUDIT.json');eng=J(D/'ENGINEERING_AUDIT.json')
assert eng['status']=='PASS_ENGINEERING_PROGRESS_NOT_FINAL'
assert a['formal_episodes']==180 and a['epochs']==1200
profile=J(R/'PROFILE_SUMMARY.json');cache=J(R/'capacity_readonly_20261110_162341/EVIDENCE.json');graph=J(R/'capacity_readonly_20261110_161909/EVIDENCE.json')
assert cache['result']['cudnn_benchmark'] is True and cache['result']['cudnn_deterministic'] is True
assert cache['exit']==0 and cache['result']['status']=='REJECT_FRAME_CACHE_NOT_BITWISE_EQUAL'
assert all(x['original_repeat']['equal'] for x in cache['result']['lengths'])
assert all(not x['sampled_action']['equal'] and x['rng_equal'] for x in cache['result']['policy'])
assert graph['exit']==1 and 'Invalid capture' in graph['log_tail']
vectors=[]
for n,v in e['light'].items():
 if 'random/evaluation/attempts_' not in n:continue
 rows=[json.loads(x) for x in v['text'].splitlines()];starts={x['group']:x['unix'] for x in rows if x['event']=='vector_attempt_start'}
 for x in rows:
  if x['event']=='vector_attempt_complete':vectors.append(dict(stage=x['stage'],task=x['task'],group=x['group'],seconds=x['unix']-starts[x['group']],steps=x['steps'],end=x['unix']))
vectors.sort(key=lambda x:x['end']);assert len(vectors)==18
obs=datetime.datetime.fromtimestamp(e['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
timing=dict(status='CONDITIONAL_ETA_NO_PRODUCTION_ACCELERATION',observed=obs,vectors=vectors,mean_vector_seconds=statistics.mean(x['seconds'] for x in vectors),remaining_object_vectors=42,linear_object_remaining_minutes=42*statistics.mean(x['seconds'] for x in vectors)/60,planning_minutes=dict(object_remaining=[150,190],goal_training_unmeasured=[100,130],goal_evaluation_unmeasured=[180,240]),total_minutes=[430,560],limits=['Only OBJECT41 measured; later task success/termination lengths differ.','Isolated probes briefly shared GPU; no uncontended optimization timing claim.','GOAL unmeasured; final CPU acceptance/comparison/publication extra.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(timing,indent=2)+'\n',encoding='utf8')
evidence={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'EVIDENCE.json',D/'ENGINEERING_AUDIT.json',R/'PROFILE_SUMMARY.json',R/'capacity_readonly_20261110_161255/EVIDENCE.json',R/'capacity_readonly_20261110_161547/EVIDENCE.json',R/'capacity_readonly_20261110_161909/EVIDENCE.json',R/'capacity_readonly_20261110_162341/EVIDENCE.json']}
decision=dict(status='DIAGNOSED_NO_SAFE_ACCELERATION_DEPLOYED',observed=obs,evidence=evidence,frame_cache='Rejected: different spatial features and sampled actions at identical CUDA RNG.',cuda_graph='Rejected as not validated: isolated capture failed in text attention-mask/CUDA allocator path; not a production failure or proof all graph variants impossible.',production_source_changes=0,production_signals=0,optimizer_calls_added=0,environment_calls_added=0,diagnostic_cost='Two cache probes, 48 full spatial encoder calls; 16 temporal and 16 head calls; 4 extra language and 4 extra vision calls. First used default benchmark flag; v2 explicitly matched production benchmark=True and deterministic=True, yielding the same differences. Graph probe completed 3 whole-path warmups then a partial failed capture. Construction/loading and partial capture work also occurred; not zero compute.',no_whole_run_speedup_claim=True)
(R/'OPTIMIZATION_DECISION.json').write_text(json.dumps(decision,indent=2)+'\n',encoding='utf8')
report=f'''# ALIGN评估慢速排查与优化核验

观察：{obs}。PARTIAL。主要差异已定位到评估代码的完整历史窗口重算；当前没有证据指向GPU故障。已实际尝试两条离线优化路径，均未达到可安全部署标准；生产未提速，不能宣称问题已完全修复。

## 当前实际执行

三个OBJECT各400轮训练及Task6–9自己的ALIGN准入完成。OBJECT41正式180/200；42/43等待，GOAL三个未开始。controller40721、评估40723和环境子进程正常。原科学/工程合同及模型、恢复文件SHA通过，160339→161707→162039全部已记录轮/诊断/正式前缀保留。旧资源与包装失败证据不覆盖。

## 具体耗时来源

`coreset_exp/eval_metric.py`每步堆叠最近8帧原始观察；`CompletePolicy.get_action → spatial_encode`每次对整窗重新编码。10条环境、2路相机、8帧稳态每步共160张图像编码，其中140张来自历史帧。之前S/q1的评估只编码新帧后拼接特征，共20张。这是8倍图像编码工作量，不是8倍整体墙钟；GPT、环境及初始化仍占时间。ALIGN训练目标没有给推理添加额外网络；本轮按冻结协议沿用Random父源的全窗评估路径。

40秒非阻塞20Hz采样799条、0采样错误、0目标信号。Python栈：视觉34.54%、时序32.67%、环境接收等待20.90%、语言2.63%、其他9.26%。CUDA异步会把等待计在后续调用，不能据此称GPT实际GPU计算占32.67%。同步离线探针在共享GPU下观察到两路视觉各0.292/0.347秒、语言0.014秒、时序约0.008–0.018秒，支持视觉计算为主要算量来源；此共享窗口不能用于报告独占性能或生产加速倍数。

采样开始正逢模型/环境切换，出现0%利用率；显存稳定后的26个样本GPU均89.35%、77–100%，约287–295W、约2370–2655MHz、60–70℃。CPU/I/O/memory pressure avg10均0。没有发现持续闲置、系统换页或本次驱动报错。此观察不能排除所有硬件问题，但不支持“GPU坏了”这一解释。

原实现600步批次约223–231秒；提前完成的148–217步批次约58–84秒。任务何时成功决定闭环墙钟。早先把S/q1的60–90分钟/套件直接套到本轮，是估时错误，已纠正。

## 两条真实优化尝试

1. **历史帧特征缓存：未通过严格等价，不部署。** 使用现有Task6示范的10条8帧窗口，无环境交互。初探使用默认benchmark标志；v2明确匹配生产benchmark=True/deterministic=True，逐项差异结论相同，两次证据均保留。原全窗重复计算各长度1–8均逐位相同；按单帧编码后缓存从长度2开始不同，特征最大绝对差最高1.84e-5。在四个已有阶段模型、相同CUDA RNG下，动作分别有54/52/54/50个分量不同（每批70），最大差8.34e-7。RNG推进相同。差异虽小，仍不能保证长闭环动作轨迹与成功率不变，故未放宽容差、未换成旧S评估器。旧S/q1还使用逐轨迹噪声与done后置零等不同执行方式，整体替换更不是等价工程修复。
2. **保留全窗口FP32计算的CUDA Graph：隔离试验捕获失败，不部署。** 当前PyTorch1.13.0/CUDA11.6路径在CLIP文本attention-mask/CUDA allocator处报Invalid capture。原错误、源码SHA与日志已保留。这里只确认这一实现未通过，不宣称所有CUDA Graph优化都不可能，也未升级/重启服务器或修改生产库。

三个诊断进程均已结束，0optimizer、0环境、0生产文件写入。存在额外离线前向计算且短暂共享GPU，具体调用账在OPTIMIZATION_DECISION；不称零成本。当前生产继续，已完成回合不重跑。没有把近似相等当作结果不变，也没有将失败的优化标为上线成功。

## 剩余时间与后续

18个已完成vector均181.28秒，OBJECT余42批线性参考126.89分钟；考虑其余seed/任务及初始化，OBJECT余150–190分钟。GOAL三训练两波仍未实测，预留100–130分钟；GOAL600正式未实测，预留180–240分钟。总余430–560分钟（约7–9.5小时），约今日23:30至明日01:40；最终CPU验收、比较与发布另计。不是固定完成保证；不根据时钟声称结束。

唯一生产完成源仍engineering_assets_resume_20261110_v2/controller.exit0且无自身进程，继续原check/audit/final路径。工程速度目录只是诊断，禁止把它当部署入口。OBJECT600完成后才GOAL，旧S/Random不重跑。当前没有通过严格等价验证的生产加速方案。
'''
(B/'EVALUATION_SPEED_DIAGNOSIS_20261110.md').write_text(report,encoding='utf8')
state=J(B/'RUN_STATUS.json');state.update(observed=obs,status='PARTIAL_RECOVERED_OBJECT_EVALUATING',snapshot=str(D),counts=a['counts'],formal_episodes=a['formal_episodes'],remaining_total_time_minutes=[430,560],speed_diagnosis=str(B/'EVALUATION_SPEED_DIAGNOSIS_20261110.md'),optimization_deployed=False)
for n in ['RUN_STATUS.json','ACTIVE_ROOT.json']:(B/n).write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
handoff=f'''## {obs} 用户要求慢速排查：已定位，安全优化尚未通过

先读EVALUATION_SPEED_DIAGNOSIS_20261110.md及engineering_eval_speed_20261110_v1/OPTIMIZATION_DECISION.json。162039实际审计PASS，OBJECT41正式180/200、其他0，三OBJECT训练完成，所有前缀与合同保持。全窗重算8帧相对S/q1单帧缓存是主要计算路径差异；GPU稳定窗口77–100%、均89.35%，未见GPU报错或系统压力。单帧缓存离线四阶段动作不逐位相等，禁止当等价切换；CUDA Graph隔离捕获失败，原失败保留，不部署任何速度工程。当前生产仍原enginev2/controller40721，未收信号未改代码，未重训重评。三个诊断已经退出，0optimizer/0环境但有前向计算及短暂资源竞争。工程速度目录不是启动入口，不重复其已有三份诊断。全队列条件余430–560分钟，CPU接受另计；无生产提速保证。最终来源和队列顺序不变。

'''
p=B/'MONITOR_HANDOFF.md';p.write_text(handoff+p.read_text(encoding='utf8'),encoding='utf8')
(B/'LATEST_PROGRESS_REPORT.md').write_text(report,encoding='utf8')
print(json.dumps(dict(status=decision['status'],formal_episodes=a['formal_episodes'],new_optimizer=0,new_environment=0)))
