"""Independent local final audit from outcome records and verified lightweight packages."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,sys,runpy,contextlib,io,datetime
R=Path(__file__).resolve().parent;D=R/sys.argv[1]
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(l) for l in p.read_text().splitlines()]
with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(R/'audit_ra_evaluation.py'),run_name='__main__')
partial=js(D/'PROGRESS_ACCEPTANCE.json');assert partial['RA_records']==200 and partial['RA_vectors_completed']==partial['RA_vectors_started']==20
assert js(D/'PACKAGE_DOWNLOAD_ACCEPTANCE.json')['status']=='PASS'
for kind in ['baseline','ra']:
 root=D/('completion_'+kind);manifest=js(root/'PACKAGE_MANIFEST.json')
 for n,h in manifest['files'].items():assert sha(root/n)==h['sha256'] and (root/n).stat().st_size==h['bytes']
 for n in ['pipeline.exit','supervisor.exit']:assert (root/n).read_text().strip()=='0'
snap=js(D/'SNAPSHOT.json');assert snap['queue_exit']==snap['wrapper_exit']=='0' and snap['failure'] is None
assert all('monitor_snapshot.py' in p for p in snap['processes'])
assert js(D/'queue/FINAL_QUEUE_ACCEPTANCE.json')['status']=='PASS'
B=D/'completion_baseline';A=D/'completion_ra'
baseaccept=js(B/'INDEPENDENT_ACCEPTANCE.json');assert baseaccept['status']=='PASS' and baseaccept['final_checkpoints']==8
assert len(baseaccept['loaded_downstream_keys'])==8 and set(baseaccept['loaded_downstream_keys'].values())=={100}
assert js(A/'PROCESS_EXIT_ACCEPTANCE.json')['status']=='PASS'
aa=js(A/'AUDIT.json');assert aa['status']=='PASS' and aa['inputs_unchanged'] and len(aa['checkpoints'])==4
assert all(x['loaded_keys']==100 for x in aa['checkpoints'])
assert js(B/'TASK6_TENSOR_PARITY.json')['equal'] and js(A/'TASK6_TENSOR_PARITY.json')['equal']
assert js(B/'random/INITIAL_STATE.json')==js(B/'coreset/INITIAL_STATE.json')==js(A/'ra_coreset/INITIAL_STATE.json')
aggregate=js(D/'queue/AGGREGATE.json');results={};matrices={};costs={};checkpoint_hashes={}
for arm,root in [('random',B),('coreset',B),('ra_coreset',A)]:
 ep=js(root/arm/'epochs.json');assert len(ep)==400 and {(x['task'],x['epoch']) for x in ep}=={(t,e) for t in range(6,10) for e in range(100)}
 assert all(x['batches']==[100,139,185,337][x['task']-6] for x in ep)
 assert sum(x['batches'] for x in ep)==ep[-1]['steps']==76100
 cp=js(root/arm/'TRAINING_COMPLETE.json');assert len(cp['checkpoints'])==4;checkpoint_hashes[arm]=cp['checkpoints']
 rates={};seconds=[];evall=root/arm/'evaluation';events=[v for p in evall.glob('attempts*.jsonl') for v in lines(p)]
 assert Counter(x['event'] for x in events)==Counter(vector_attempt_start=20,vector_attempt_complete=20,environment_construct_start=10,environment_construct_complete=10,environment_closed=10)
 for s in range(6,10):
  for t in range(6,s+1):
   rr=lines(evall/f'trials_stage{s}_task{t}.jsonl');assert len(rr)==20 and {x['trial'] for x in rr}==set(range(20))
   for x in rr:
    assert x['seed']==41 and x['stage']==s and x['task']==t and x['arm']==arm and x['epoch']==99 and type(x['success']) is bool
    assert x['checkpoint_sha256']==cp['checkpoints'][f'mlrifa_{s}_model_ep99.pth']
   for g in [0,1]:
    subset=[x for x in events if (x.get('stage'),x.get('task'),x.get('group'))==(s,t,g)]
    start=[x for x in subset if x['event']=='vector_attempt_start'];end=[x for x in subset if x['event']=='vector_attempt_complete']
    assert len(start)==len(end)==1 and end[0]['env_seed']==41+g
    assert end[0]['successes']==[x['success'] for x in sorted(rr,key=lambda v:v['trial'])][g*10:(g+1)*10]
    seconds.append(end[0]['unix']-start[0]['unix'])
   rates[s,t]=F(sum(x['success'] for x in rr),20)
 m=dict(AUC=sum(sum(rates[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(rates[t,t] for t in range(6,10))/4,NBT=sum(sum(rates[t,t]-rates[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(rates[9,t] for t in range(6,9))/3,FinalAll=sum(rates[9,t] for t in range(6,10))/4)
 assert all(v==F(aggregate['metrics'][arm][k]['fraction'])==F(aa['metrics'][arm][k]['fraction']) for k,v in m.items())
 results[arm]={k:dict(fraction=str(v),percent=float(v*100)) for k,v in m.items()}
 matrices[arm]=[[float(rates[s,t]*100) if t<=s else None for t in range(6,10)] for s in range(6,10)]
 costs[arm]=dict(training_minutes=cp['seconds_this_invocation']/60,active_vector_minutes=sum(seconds)/60)
for x in aa['checkpoints']:assert x['sha256']==checkpoint_hashes['ra_coreset'][f"mlrifa_{x['task']}_model_ep99.pth"]
deltas={arm:{k:dict(fraction=str(F(results['ra_coreset'][k]['fraction'])-F(v['fraction'])),pp=float(100*(F(results['ra_coreset'][k]['fraction'])-F(v['fraction'])))) for k,v in results[arm].items()} for arm in ['random','coreset']}
out=dict(status='PASS',experiment_status='COMPLETED',seed=41,epochs=1200,optimizer_steps=228300,final_checkpoints_strictly_loaded=12,formal_records=600,vector_attempts=60,technical_exceptions=0,queue_exit=0,wrapper_exit=0,all_pipeline_supervisor_exit=0,no_residual_scientific_processes=True,task6_actual_tensor_and_initial_rng_equal=True,source_input_package_hashes_pass=True,storage=partial['storage'],metrics=results,matrices=matrices,RA_minus=deltas,costs=costs,observed_unix=snap['time'],auto_extend=False)
(D/'LOCAL_FINAL_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n')
status=dict(status='COMPLETED',time=datetime.datetime.now().astimezone().isoformat(),snapshot=D.name,epochs=1200,steps=228300,records=600,expected_records=600,final_acceptance='PASS',auto_extend=False)
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
labels={'random':'Random + IFA','coreset':'Coreset + IFA','ra_coreset':'RA-Coreset + IFA'}
metric_rows='\n'.join('| '+labels[arm]+' | '+' | '.join(f"{m[k]['percent']:.6f}" for k in ['AUC','FWT','NBT','OldFinal','FinalAll'])+' |' for arm,m in results.items())
matrix_rows='\n'.join(f"| {s} | {t} | "+' | '.join(f'{matrices[arm][s-6][t-6]:.0f}' for arm in labels)+' |' for s in range(6,10) for t in range(6,s+1))
cost_rows='\n'.join(f"| {labels[arm]} | {v['training_minutes']:.2f} | {v['active_vector_minutes']:.2f} |" for arm,v in costs.items())
completed_time=datetime.datetime.fromtimestamp(js(D/'queue/FINAL_QUEUE_ACCEPTANCE.json')['time'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
report=f'''# GOAL seed41 三组最终验收

**COMPLETED**。服务器于 {completed_time} 通过最终队列验收；本地重新读取600条正式episode、60组vector尝试，并用Fraction独立复算全部指标，LOCAL_FINAL_ACCEPTANCE为PASS。三组各400完整epoch、76100步，共1200epoch/228300步；12个最终ep99 checkpoint严格加载凭证均为100个下游键。队列、wrapper和所有pipeline/supervisor退出0，无残留科学进程。无需继续等待，没有追加训练或评估。

以下均为百分数，NBT越低越好，其他列越高越好。精确分数保存在LOCAL_FINAL_ACCEPTANCE.json。

| 方法 | AUC | FWT | NBT | 最终旧任务均值 | 最终全部任务均值 |
|---|---:|---:|---:|---:|---:|
{metric_rows}

RA相对Coreset：AUC +6.875pp，FWT +3.75pp，NBT -6.666667pp，最终旧任务均值 +6.666667pp，最终全部任务均值 +8.75pp。相对Random：AUC +9.6875pp，FWT +11.25pp，NBT +2.777778pp（遗忘指标更差），最终旧任务均值 +6.666667pp，最终全部任务均值 +10pp。不将AUC改善等同于所有任务和所有遗忘指标均改善。

每格20个共同编号初态，全部有效单元如下；未来任务未评估。

| 训练阶段 | 评估任务 | Random | Coreset | RA-Coreset |
|---:|---:|---:|---:|---:|
{matrix_rows}

RA的最终Task6/7/8/9为35/100/85/85%。相比Coreset，分别+10/0/+10/+15pp；相比Random，分别-5/0/+25/+20pp。Task6遗忘仍未解决：从刚学完70%降到35%；阶段8的Task6也只有35%，低于Random的55%。Task7三组均为100%，不能从这个饱和值判断谁更好。

验收与可追溯证据：

- 新seed41官方GOAL权重起步、state=True、每任务100完整epoch/batch10、BC+0.1平均IFA/gamma0.7/warmup500逐step线性及原排序shuffle。原0–19初态、10lane、env41/42、600步与原GMM采样，阶段global RNG重置41不变。
- Task6三组实际模型张量相同，初始化模型/RNG一致；不是用含路径元数据的整checkpoint文件哈希冒充模型等价。
- RA任务6–9配额499/445/452/985、硬锚点249/222/226/492、交换119/107/112/244。实际有序成员与离线结果完全一致，源与buffer载荷哈希通过。
- 源码/配置/模态/父输入冻结合同通过；下载两份PACKAGE_MANIFEST逐文件验证。全部600条正式记录和60个vector start/complete一一匹配，无技术异常、无重试。预检与正式结果分开。
- 18项基线软件和8项RA软件已通过；各4次隔离GPU更新恢复此前已通过，本次不重跑。最终检查点实际加载由原流水线完成，本地下载仅元数据、日志和结果，不含权重、latent、buffer载荷、原actions或RNG二进制。

| 方法 | 正式训练分钟 | vector执行分钟之和 |
|---|---:|---:|
{cost_rows}

vector时长不含全部环境初始化、进程启动和审计开销，不能充当完整总耗时。完整命令/时序在各包commands与timings记录中。

本次真实命令（均退出0）：

1. `python execution/20261019_ra_coreset_goal_seed41/check_progress.py`
2. `python execution/20261019_ra_coreset_goal_seed41/collect_ra_progress.py {D.name}`
3. `python execution/20261019_ra_coreset_goal_seed41/collect_completion.py {D.name}`，仅调用远端元数据打包器和SFTP；每条远端命令及退出码见PACKAGE_DOWNLOAD_ACCEPTANCE.json。
4. `python execution/20261019_ra_coreset_goal_seed41/audit_completion.py {D.name}`，只读复算，无模型/环境调用。该入口复用旧进度审计时生成的PROGRESS_ACCEPTANCE仍写PARTIAL，最终状态以其后的LOCAL_FINAL_ACCEPTANCE和RUN_STATUS为准。

服务器产物分别保留在 `/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019` 和 `/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed41_20261019`。本地轻量证据位于本报告同目录completion_baseline和completion_ra，完整发布名单与哈希见PUBLICATION_MANIFEST；推送SHA在工作区PUBLICATION_RECEIPT中记录。旧seed42–44结果未替换。

此seed是看到42–44结果后的探索性扩展；只有一个新增训练seed、每单元20次闭环，动作随机流没有逐轨迹隔离。成员改变会改变文件排序下的batch内容及IFA实际输入，不能把差异归因于单独的覆盖半径或纯锚点作用。官方pretrain state元数据仍unknown，调度/最终ep99规则包含工程补全，不能声称精确恢复作者轨迹。RA达到本轮报告门槛，不自动扩展其他seed/套件或启动下一轮；交付发布后删除goal-seed41定时任务。
'''
(D/'REPORT.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(status='PASS',metrics=results,RA_minus=deltas,costs=costs),indent=2))
