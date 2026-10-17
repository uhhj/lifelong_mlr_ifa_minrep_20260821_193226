"""Independent read-only acceptance and dual-comparator reporting; no ML imports."""
import hashlib,json,statistics,importlib.util
from fractions import Fraction as F
from pathlib import Path
R=Path(__file__).resolve().parent
E=R/'final_evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,seed):
 c={}; records={}
 for s in range(6,10):
  for t in range(6,s+1):
   rows=lines(p/f'trials_stage{s}_task{t}.jsonl')
   assert len(rows)==20 and {r['initial_state_index'] for r in rows}==set(range(20))
   for r in rows:
    assert (r['seed'],r['stage'],r['task'],r['epoch'])==(seed,s,t,99) and type(r['success']) is bool
    records[s,t,r['initial_state_index']]=r
   c[s,t]=F(sum(r['success'] for r in rows),20)
 return c,records
def metrics(r):
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
manifest=read(R/'final_manifest.json')
assert all(sha(E/p)==v['sha256'] for p,v in manifest.items())
hashes=read(E/'hash_checks.json')
assert hashes['input_mismatches']==[] and hashes['orchestration_unchanged']
assert all(not v['source_mismatches'] and not v['config_mismatches'] for v in hashes['seeds'].values())
assert read(E/'queue/FINAL_QUEUE_ACCEPTANCE.json')['no_residual_process']
assert (E/'queue/queue.exit').read_text().strip()=='0'
# Re-run the existing complete seed42 read-only acceptance, including source manifest.
p42=R.parent/'20261017_zr_coreset_seed42'
spec=importlib.util.spec_from_file_location('seed42_audit',p42/'independent_local_acceptance.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main()
new={42:load(p42/'final_evidence/run/coreset_ifa_zr/evaluation',42)}
first={42:load(p42/'final_evidence/baseline',42)}
audits={}; costs={}
cost42=p42/'final_acceptance/run/COST.json'
if cost42.exists():costs[42]=read(cost42)
for seed in (43,44):
 o=E/f'seed{seed}'; a=o/'coreset_ifa_zr'
 n,nr=load(a/'evaluation',seed); b,br=load(R/f'launch_evidence/seed{seed}/baseline',seed)
 new[seed]=(n,nr);first[seed]=(b,br)
 for k,r in nr.items():
  for f in ('initial_state_sha256','modal_sha256'):assert r[f]==br[k][f]
 epochs=read(a/'epochs.json');assert len(epochs)==400
 assert {(x['task'],x['epoch']) for x in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
 total=0
 for x in epochs:
  assert x['batches']=={6:157,7:227,8:309,9:362}[x['task']]
  total+=x['batches'];assert total==x['steps']
 assert total==105500
 for s,t in n:
  attempts=lines(a/'evaluation'/f'attempts_stage{s}_task{t}.jsonl')
  assert not any(v['event']=='technical_exception' for v in attempts)
  starts=[v for v in attempts if v['event']=='vector_attempt_start'];ends=[v for v in attempts if v['event']=='vector_attempt_complete']
  assert len(starts)==len(ends)==2
  for g in (0,1):
   st=next(v for v in starts if v['group']==g);en=next(v for v in ends if v['group']==g)
   assert st['env_seed']==en['env_seed']==seed+g
   assert st['initial_indices']==list(range(10*g,10*g+10)) and st['unix']<=en['unix'] and en['steps']<=600
   for i,v in enumerate(en['successes']):assert v==nr[s,t,10*g+i]['success']
 checks=lines(o/'DIRECT_CHANNEL_ISOLATION.jsonl');assert len(checks)==44 and all(x['passed'] for x in checks)
 for task,size in {6:0,7:781,8:1521,9:2301}.items():
  cc=[x for x in checks if x['stage']==task]
  assert {x['epoch'] for x in cc}=={-1,9,19,29,39,49,59,69,79,89,99}
  end=next(x for x in cc if x['epoch']==99)
  assert end['full_legal_old_buffer'] and sum(x['n'] for x in end['rows'] if x['kind']=='legal_buffer')==size
 rr=[x for c in checks for x in c['rows']]
 assert all(x['passed'] and x['selected_exact'] for x in rr)
 assert max(x['old_gate'] for x in rr)<=1e-6 and max(x['contribution'] for x in rr)<=1e-6
 assert all(x['old_unchanged'] for x in lines(o/'FROZEN_BRANCH_IDENTITY.jsonl'))
 cp=read(o/'CHECKPOINT_ACCEPTANCE.json');assert cp['status']=='PASS' and len(cp['checkpoints'])==4
 for c in cp['checkpoints']:
  assert c['branches']==[str(t) for t in range(6,c['stage']+1)]
  assert all(v=='torch.float64' for v in c['gate_dtypes'].values())
  assert all(r['checkpoint_sha256']==c['sha256'] for k,r in nr.items() if k[0]==c['stage'])
 audit=read(o/'FINAL_AUDIT.json')
 for k in ('inputs_unchanged','buffer_members_order_payloads_equal','IFA_identity_equal','structural_checks_pass'):assert audit[k]
 proc=read(o/'PROCESS_EXIT_ACCEPTANCE.json');assert proc['status']=='PASS' and proc['remaining']==[]
 assert read(o/'RUN_STATUS.json')['status']=='COMPLETED'
 assert (o/'pipeline.exit').read_text().strip()==(o/'supervisor.exit').read_text().strip()=='0'
 saved=read(o/'METRIC_COMPARISON.json')
 for k,v in metrics(n).items():assert F(saved['new'][k]['fraction'])==v
 for k,v in metrics(b).items():assert F(saved['baseline'][k]['fraction'])==v
 costs[seed]=read(o/'COST.json')
 audits[seed]={'epochs':400,'steps':105500,'checkpoints':4,'episodes':200,'vector_attempts':20,'structural_checks':44,'old_gate_max':max(x['old_gate'] for x in rr),'contribution_max':max(x['contribution'] for x in rr),'status':'PASS'}
primary=dict(first)
primary[43]=load(R.parent/'20261016_seed43_three_arm_repeat_v1/completion_evidence/RC/coreset/evaluation',43)
for k,r in primary[43][1].items():
 for f in ('initial_state_sha256','modal_sha256'):assert r[f]==new[43][1][k][f]
def summarize(baselines):
 rows=[]
 for s in (42,43,44):
  bm=metrics(baselines[s][0]);nm=metrics(new[s][0])
  gates=dict(AUC=nm['AUC']>bm['AUC'],FWT=nm['FWT']>=bm['FWT'],NBT=nm['NBT']<bm['NBT'],OldFinal=nm['OldFinal']>=bm['OldFinal'])
  rows.append(dict(seed=s,baseline={k:str(v) for k,v in bm.items()},new={k:str(v) for k,v in nm.items()},delta_pp={k:float(100*(nm[k]-v)) for k,v in bm.items()},gates=gates,all_gates=all(gates.values())))
 stats={}
 for k in metrics(new[42][0]):
  stats[k]={}
  for arm in ('baseline','new','delta_pp'):
   v=[r[arm][k] if arm=='delta_pp' else float(F(r[arm][k])*100) for r in rows]
   stats[k][arm]={'mean':statistics.mean(v),'sample_sd':statistics.stdev(v)}
 return {'rows':rows,'statistics':stats}
summary={'primary_repeat43':summarize(primary),'sensitivity_first43':summarize(first)}
summary['matrices_percent']={str(s):{name:{f'{stage},{task}':float(v*100) for (stage,task),v in src[s][0].items()} for name,src in [('primary_baseline',primary),('first_baseline',first),('ZR',new)]} for s in (42,43,44)}
summary['cost']=costs
summary['post_result_seed43_comparator_change']=True
(R/'FINAL_RESULTS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert all(sha(E/p)==v['sha256'] for p,v in manifest.items())
out={'status':'COMPLETED','local_independent_acceptance':'PASS','seeds':[42,43,44],'total_epochs':1200,'total_steps':316500,'strict_final_checkpoints':12,'formal_episodes':600,'vector_attempts':60,'technical_exception_attempts':0,'structural_checks':132,'extension_audits':audits,'source_config_input_checks':'PASS','unique_extension_inputs':hashes['unique_inputs'],'queue_exit':0,'input_files_unchanged':True,'no_residual_process':True,'new_rollouts_or_training_in_this_audit':False,'seed42_original_failure_preserved':True}
(R/'LOCAL_FINAL_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
report=['# Coreset＋IFA＋指令零响应残差：三seed最终结果','',
        '状态：COMPLETED；独立只读验收PASS。seed43/44队列于2026-10-17 22:33（北京时间）退出0。三seed共1200epoch、316500步、12个严格加载最终checkpoint、600条正式闭环和60次向量尝试，未发现技术异常重试。132次结构检查全部通过。',
        '', '按用户最新要求，seed43主比较使用第二次完整重跑Coreset＋IFA（repeat1），seed42/44沿用首次。首次43仍完整保留辅助表；历史五seed表未覆盖。这是在看到结果后调整比较口径，不能描述为预注册比较。训练缓存/成员/IFA/初始化身份仍绑定原冻结父运行，未改合同或运行结果。', '',
        '## 主结果（百分制）','', '|Seed|基线AUC|ZR AUC|ΔAUC pp|基线FWT|ZR FWT|基线NBT|ZR NBT|', '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in summary['primary_repeat43']['rows']:
 vals=[float(F(r[a][k])*100) for k in ('AUC','FWT','NBT') for a in ('baseline','new')]
 report.append(f"|{r['seed']}|{vals[0]:.4f}|{vals[1]:.4f}|{r['delta_pp']['AUC']:+.4f}|{vals[2]:.4f}|{vals[3]:.4f}|{vals[4]:.4f}|{vals[5]:.4f}|")
report+=['','|指标|基线均值±样本SD|ZR均值±样本SD|配对差均值±样本SD（pp）|','|---|---:|---:|---:|']
for k,stat in summary['primary_repeat43']['statistics'].items():
 report.append('|'+k+'|'+'|'.join(f"{stat[a]['mean']:.4f} ± {stat[a]['sample_sd']:.4f}" for a in ('baseline','new','delta_pp'))+'|')
report+=['','NBT沿用原有符号定义，越小越好；不能单凭负NBT推断绝对旧任务成功率。OldFinal是最终阶段旧任务6–8均值，FinalAll是最终四任务均值。', '',
         '## 全部任务矩阵（成功率%）','', '|Seed|阶段|主基线 T6/T7/T8/T9|ZR T6/T7/T8/T9|','|---|---|---|---|']
for seed in (42,43,44):
 for st in range(6,10):
  cells=[]
  for src in (primary,new):cells.append(' / '.join(f'{float(src[seed][0][st,t]*100):g}' if t<=st else '—' for t in range(6,10)))
  report.append(f'|{seed}|{st}|{cells[0]}|{cells[1]}|')
report+=['', '## 首次43敏感性对照','', '|指标|首次基线三seed均值±SD|ZR均值±SD|配对差均值±SD（pp）|','|---|---:|---:|---:|']
for k,stat in summary['sensitivity_first43']['statistics'].items():
 report.append('|'+k+'|'+'|'.join(f"{stat[a]['mean']:.4f} ± {stat[a]['sample_sd']:.4f}" for a in ('baseline','new','delta_pp'))+'|')
report+=['','首次43的AUC94.375%，对应ZR73.4375%，差−20.9375pp；第一次高分未被判作无效。原始全矩阵见FINAL_RESULTS.json（first_baseline）。', '',
         '## 结构与预定筛选门槛','', '三个seed均满足新增分支对旧指令的直接零响应阈值、旧分支参数冻结与严格加载检查。任务7/8最终全旧buffer扫描分别覆盖781/1521条，任务9覆盖2301条。但共享时间网络和策略头仍会变化，因此结构零响应不等价于整个旧策略不遗忘。', '',
         '|Seed|AUC上升|FWT不降|NBT下降|OldFinal不降|全部筛选条件|','|---|---|---|---|---|---|']
for r in summary['primary_repeat43']['rows']:
 report.append('|'+str(r['seed'])+'|'+'|'.join('通过' if r['gates'][k] else '未通过' for k in ('AUC','FWT','NBT','OldFinal'))+'|'+('通过' if r['all_gates'] else '未通过')+'|')
report+=['', '工程验收通过与科学筛选通过是两回事。seed42正收益较大，43负收益，44温和正收益且FWT下降1.25pp；不能据此称为稳定互补模块。没有自动扩展或调参。', '', '## 成本与执行证据','', '|Seed|训练分钟|评估分钟|结构诊断秒|','|---|---:|---:|---:|']
for s,c in costs.items():report.append(f"|{s}|{c['training']['seconds']/60:.2f}|{c['evaluation']['seconds']/60:.2f}|{c['structural_diagnostic_seconds']:.2f}|")
report+=['', '每组额外FP32残差参数18784；轻量证据包204文件，约3.42MB；未下载权重/latent/演示/buffer载荷。18055个去重原输入以及冻结源代码/配置/队列哈希通过。成员顺序载荷与IFA身份仍按冻结原父组验收。seed43 repeat1没有重新进行完整存储/IFA逐张量配对审计，不能把修订比较当严格同轨迹干预。', '',
         '真实执行与退出码：Q/run_queue.sh退出0；seed43与44的train/evaluate/finalize/pipeline/supervisor均退出0；Q/aggregate_three_seeds.py退出0；Q/final_queue_audit.py退出0；monitor/package_final_readonly.py退出0；本地python -X utf8 execution/20261017_zr_coreset_extension/final_local_audit.py退出0。Q=/root/mlr_ifa/experiment_workspaces/ZR_coreset_extension_20261017。seed42原始失败退出1和外置恢复退出0均保留，不将旧失败改写为0。', '',
         '两个新增运行与checkpoint保存在/root/mlr_ifa/experiments/ZR_coreset_OBJECT_seed43_20261017及seed44同名目录；已有seed42在/data/mlr_ifa/experiments/ZR_coreset_OBJECT_seed42_20261017。本地final_evidence、final_manifest.json、LOCAL_FINAL_ACCEPTANCE.json与FINAL_RESULTS.json可追踪记录、模型哈希、退出证据及复算。', '',
         '边界：仅三个探索性seed；是在看到42后扩展，43主比较又作事后调整。repeat1基线用RTX4090，当前ZR/首次基线用RTX3090；原全局动作RNG没有逐轨迹隔离。官方pretrain state模态元数据未知，学习率调度有公开细节补全，模型新增参数、共享遗忘仍存在。不能声称统计显著、普遍因果或恢复作者精确轨迹。未训练或追加闭环，未push，未启动下一轮。']
(R/'FINAL_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
print(json.dumps({'acceptance':out,'primary_statistics':summary['primary_repeat43']['statistics'],'primary_rows':summary['primary_repeat43']['rows']},indent=2))
