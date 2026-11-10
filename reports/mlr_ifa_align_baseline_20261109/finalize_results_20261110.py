"""Local accepted-outcome recomputation and report; no remote/scientific execution."""
from pathlib import Path
from fractions import Fraction as F
import csv, datetime, hashlib, json, statistics, sys

R=Path(__file__).resolve().parent
W=R.parents[1]
D=Path(sys.argv[1]).resolve()
PRE=R/'engineering_cached_eval_20261110_v1/check_readonly_20261110_194749'
J=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=J(D/'LOCAL_FINAL_AUDIT.json'); c=J(D/'BASELINE_COMPARISON.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
assert c['source_final_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json')
before=J(PRE/'EVIDENCE.json'); after=J(D/'EVIDENCE.json')
assert not after['processes']
preserved=[]
for n,v in before['light'].items():
    assert n in after['light'] and v==after['light'][n],n
    preserved.append(n)
assert after['light']['cached_engineering/controller.exit']['text'].strip()=='0'
completion=json.loads(after['light']['cached_engineering/STATUS.json']['text'])
assert completion['status']=='NATURAL_COMPLETE_PENDING_CPU_ACCEPTANCE'
now=datetime.datetime.now().astimezone().isoformat()
finished=datetime.datetime.fromtimestamp(completion['observed']).astimezone().isoformat()
metrics=['AUC','FWT','NBT','OldFinal','NewFinal','FinalAll']
def calc(matrix):
    m={tuple(map(int,k.split('_'))):F(str(v)) for k,v in matrix.items()}
    assert set(m)=={(s,t) for s in range(6,10) for t in range(6,s+1)}
    return {k:float(v) for k,v in dict(
        AUC=sum(sum(m[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,
        FWT=sum(m[t,t] for t in range(6,10))/4,
        NBT=sum(sum(m[t,t]-m[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
        OldFinal=sum(m[9,t] for t in range(6,9))/3,NewFinal=m[9,9],
        FinalAll=sum(m[9,t] for t in range(6,10))/4).items()}
rows=[]; bindings={}
for suite in ['OBJECT','GOAL']:
    p=W/'execution/20261025_ccbp_v4_full'/f'{suite}_THREE_SEED_BASELINES.json'
    historical=J(p); assert historical['status']=='PASS_LOCAL_READONLY_RECOMPUTATION'
    bindings[str(p.relative_to(W))]=sha(p)
    for name,h in historical['source_sha256'].items():
        source=Path(name)
        assert source.exists() and sha(source)==h, str(source)
    for old in historical['runs']:
        m=calc(old['matrix'])
        assert all(abs(m[k]-v['percent'])<1e-9 for k,v in old['metrics'].items())
        if old['method']=='MLR_IFA':
            bound=c['cases'][f"{suite}_{old['seed']}"]['baseline_metrics']
            assert all(abs(m[k]-bound[k]['percent'])<1e-9 for k in metrics)
        rows.append(dict(suite=suite,seed=old['seed'],method=old['method'],metrics=m,matrix=old['matrix']))
    for seed in [41,42,43]:
        r=a['cases'][f'{suite}_{seed}']; m=calc(r['matrix'])
        assert all(abs(m[k]-r['metrics'][k]['percent'])<1e-9 for k in metrics)
        rows.append(dict(suite=suite,seed=seed,method='MLR_IFA_ALIGN',metrics=m,matrix=r['matrix']))
summary={}; effects={}
for suite in ['OBJECT','GOAL']:
    summary[suite]={}
    for method in ['MLR_IFA','MLR_IFA_ALIGN','RA','RA_ALIGN']:
        group=[r for r in rows if r['suite']==suite and r['method']==method]
        assert sorted(r['seed'] for r in group)==[41,42,43]
        summary[suite][method]={k:dict(mean=statistics.mean(r['metrics'][k] for r in group),sample_sd=statistics.stdev(r['metrics'][k] for r in group)) for k in metrics}
    s=summary[suite]
    effects[suite]={k:dict(align_under_random=s['MLR_IFA_ALIGN'][k]['mean']-s['MLR_IFA'][k]['mean'],
        align_under_ra=s['RA_ALIGN'][k]['mean']-s['RA'][k]['mean'],
        ra_under_align=s['RA_ALIGN'][k]['mean']-s['MLR_IFA_ALIGN'][k]['mean'],
        descriptive_interaction=(s['RA_ALIGN'][k]['mean']-s['RA'][k]['mean'])-(s['MLR_IFA_ALIGN'][k]['mean']-s['MLR_IFA'][k]['mean'])) for k in metrics}
limits=c['limits']+[
    'Historical RA/RA+ALIGN matrices are read-only accepted results; the 2x2 interaction is descriptive, not a controlled same-execution causal estimate.',
    'S uses per-trajectory paired Normal, done-action zeroing and other execution differences; this ALIGN versus Random result does not prove evaluator equality with S.',
    'CPU acceptance verifies stored models, recovery, identities and recorded budget/outcomes; it does not repeat rollouts or prove bitwise cached closed-loop equality.',
    'No new training, completed-rollout repetition or baseline reevaluation was performed to resolve these limitations.'
]
out=dict(status='PASS_ACCEPTED_OUTCOME_REPORT_RECOMPUTATION',observed=now,rows=rows,summary=summary,
         descriptive_effects_pp=effects,source_bindings=bindings,acceptance_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),limits=limits)
write(R/'FULL_RESULTS.json',out)
with (R/'FULL_METRICS.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['suite','seed','method',*metrics],lineterminator='\n');w.writeheader()
    for r in rows:w.writerow({k:r[k] for k in ['suite','seed','method']}|r['metrics'])
labels={'MLR_IFA':'MLR+IFA（Random）','MLR_IFA_ALIGN':'MLR+IFA+ALIGN（本轮Random）','RA':'MLR+IFA+RA','RA_ALIGN':'MLR+IFA+RA+ALIGN'}
lines=['# MLR+IFA+ALIGN 两套件最终结果','',f'六run实际自然结束：{finished}。报告生成：{now}。',
       '','状态：PASS_COMPLETE_LOCAL_ACCEPTANCE。六run各400轮、200正式回合；24阶段ALIGN准入均通过。',
       '实际远端CPU检查24个阶段模型、6份完整恢复状态、60个评估单元的初态和保存随机状态；本地复算1200回合。544800生产更新、120隔离准入更新、0训练期环境采集，6次正式评估互斥完整。验收新增0optimizer、0环境回合。',
       f'验收前后{len(preserved)}份既有轻量证据逐项不变；原故障、原退出1、恢复与缓存交接证据均保留。',
       '','## 与原Random基线的完整指标','',
       '三个seed（41、42、43）的均值±样本标准差，单位%。NBT越低越好，其余越高越好。',
       '','|套件|方案|AUC|FWT|NBT|OldFinal|NewFinal|FinalAll|','|---|---|---:|---:|---:|---:|---:|---:|']
for suite in summary:
    for method in ['MLR_IFA','MLR_IFA_ALIGN']:
        lines.append('|'+suite+'|'+labels[method]+'|'+'|'.join(f"{summary[suite][method][k]['mean']:.4f} ± {summary[suite][method][k]['sample_sd']:.4f}" for k in metrics)+'|')
lines+=['','## 三seed的完整指标与配对结果','','SF为本轮成功/旧Random失败；FS为本轮失败/旧Random成功。仅初态身份配对，不声称两次动作噪声逐轨迹完全相同。',
        '','|套件/seed|方案|AUC|FWT|NBT|OldFinal|NewFinal|FinalAll|','|---|---|---:|---:|---:|---:|---:|---:|']
for case,r in c['cases'].items():
    for typ,label in [('baseline_metrics','旧Random'),('new_metrics','本轮Random+ALIGN')]:
        lines.append('|'+case+'|'+label+'|'+'|'.join(f"{r[typ][k]['percent']:.4f}" for k in metrics)+'|')
lines+=['','|Case|SF|FS|AUC差值(pp)|FinalAll差值(pp)|','|---|---:|---:|---:|---:|']
for case,r in c['cases'].items():lines.append(f"|{case}|{r['new_success_baseline_failure']}|{r['new_failure_baseline_success']}|{r['delta_pp']['AUC']:+.4f}|{r['delta_pp']['FinalAll']:+.4f}|")
lines+=['','## 结果解释','',
 'OBJECT：AUC提高17.9861pp，三个seed分别提高18.5417/17.2917/18.1250pp；FinalAll提高16.6667pp，三个seed均提高。新任务最终成功率提高35pp，旧任务最终均值提高10.5556pp。NBT均值保持−2.7778，不能称所有指标均改善。600回合SF=138、FS=41。',
 'GOAL：AUC平均提高7.4653pp，三个seed分别提高19.3750/1.5625/1.4583pp。FinalAll均值提高6.6667pp，但逐seed为+26.25/−1.25/−5.00pp，均值收益主要来自seed41；不能称三个seed最终成功率一致改善。OldFinal均值提高5.5556pp，但seed42/43分别下降1.6667/8.3333pp。NBT均值下降0.8333pp，但seed43反而增加9.1667pp。600回合SF=92、FS=57。',
 '这些已观测结果支持在Random选样配置中加入ALIGN能提高两套件的平均AUC，其中OBJECT方向更一致；不据此声称统计显著、跨任务普遍收益或所有差异均由ALIGN单一因素造成。',
 '','## 与既有RA配置组成的描述性2×2','',
 '|套件|方案|AUC|FWT|NBT|OldFinal|NewFinal|FinalAll|','|---|---|---:|---:|---:|---:|---:|---:|']
for suite in summary:
    for method in labels:
        lines.append('|'+suite+'|'+labels[method]+'|'+'|'.join(f"{summary[suite][method][k]['mean']:.4f} ± {summary[suite][method][k]['sample_sd']:.4f}" for k in metrics)+'|')
lines+=['','|套件|指标|Random中加入ALIGN|RA中加入ALIGN|ALIGN中加入RA|描述性交互差值|','|---|---|---:|---:|---:|---:|']
for suite in effects:
    for k in metrics:lines.append('|'+suite+'|'+k+'|'+'|'.join(f'{v:+.4f}' for v in effects[suite][k].values())+'|')
lines+=['','本轮Random+ALIGN的平均AUC与FinalAll在两个套件均高于既有RA+ALIGN。这不支持“RA与ALIGN在所有配置中单调叠加增益”的写法；历史执行条件不完全相同，表中的交互量只是既有结果的差分，不是受控因果交互检验。',
 '','## 全部低于旧Random的任务单元','','|Case|训练stage_评估task|差值(pp)|','|---|---|---:|']
for n in c['all_negative_cells']:lines.append(f"|{n['case']}|{n['cell']}|{n['delta_pp']:+.4f}|")
lines+=['','## 完整十单元矩阵','','|Case|Stage|Task|旧Random|本轮Random+ALIGN|','|---|---:|---:|---:|---:|']
binding=J(R/'BASELINE_BINDING.json')
for case,r in a['cases'].items():
    for key,v in r['matrix'].items():
        s,t=key.split('_');lines.append(f"|{case}|{s}|{t}|{binding['cases'][case]['baseline_matrix'][key]:.1f}|{v:.1f}|")
lines+=['','## 执行边界与保留记录','',
 'OBJECT41/42保留原全窗评估；OBJECT43及GOAL41/42/43使用经每个实际加载模型准入的单帧缓存。缓存与全窗比较容差atol=rtol=1e-4，40个实际模型加载准入通过；不能把容差通过写成闭环逐位一致。原global Normal、done处理及环境生命周期保持本轮Random父源口径。',
 '原MLR+IFA与完整S的评估实现存在已查明差异，详见TWO_ROUND_PROTOCOL_AUDIT_20261110.md。本报告不把它们称为严格同实现比较。跨历史硬件、global随机流、开发集反复使用与预训练来源未闭合的限制均保留。',
 '既有部署/恢复工程失败及速度诊断原记录完整保留。离线缓存/速度探查没有optimizer或环境回合，但有前向计算成本；不把它们写成零计算。没有为统一执行口径重复任何已完成训练或正式评估，旧基线/S/q1结果均只读。',
 '',f'最终CPU证据：{D.relative_to(R).as_posix()}。FULL_RESULTS.json记录来源SHA及2×2复算；FULL_METRICS.csv提供24组方案/seed的六指标。']
(R/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
accept=dict(status='PASS_COMPLETE_LOCAL_ACCEPTANCE',observed=now,natural_completion=finished,
 final_directory=D.relative_to(R).as_posix(),local_final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),
 comparison_sha256=sha(D/'BASELINE_COMPARISON.json'),report_sha256=sha(R/'FINAL_REPORT.md'),
 full_results_sha256=sha(R/'FULL_RESULTS.json'),epochs=2400,production_optimizer=544800,
 isolated_admission_optimizer=120,formal_episodes=1200,training_environments=0,
 checkpoints_cpu=24,recoveries_cpu=6,new_acceptance_optimizer=0,new_acceptance_environments=0,
 existing_light_files_preserved=len(preserved),cache_boundary=a['evaluation_cache_boundary'],
 no_completed_training_or_rollout_repeated=True,publication='VERIFY_WITH_LOCAL_PUBLICATION_RECEIPT',limits=limits)
write(R/'ACCEPTANCE.json',accept)
for name in ['RUN_STATUS.json','ACTIVE_ROOT.json']:
    state=J(R/name);state.update(status='COMPLETE_ACCEPTED',observed=now,
        controller_active=False,final_wrapper_status='ACTUAL_PASS_COMPLETE_LOCAL_ACCEPTANCE',
        final_directory=accept['final_directory'],acceptance='ACCEPTANCE.json',report='FINAL_REPORT.md',
        snapshot=str(D),evidence_sha256=sha(D/'EVIDENCE.json'))
    write(R/name,state)
handoff=(f'## {now} 六run实际完成并接受\n\n自然结束{finished}。24CP/6完整恢复/1200初态及保存RNG、预算与6次评估环境互斥均实际通过；本次CPU与本地复算0optimizer/0环境，所有既有轻量证据不变。最终证据{accept["final_directory"]}。\n\nACCEPTANCE.json、FINAL_REPORT.md、FULL_RESULTS.json与FULL_METRICS.csv已生成；本轮发布状态以PUBLICATION_RECEIPT.json及远端提交覆盖这些文件为准。全部科学工作结束，不再运行controller/部署/训练/正式评估/最终CPU入口。用户明确不重跑，旧失败和比较执行差异均保留；此前PARTIAL记录只作历史。\n\n')
path=R/'MONITOR_HANDOFF.md';path.write_text(handoff+path.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(dict(status=accept['status'],completion=finished,summary=summary,effects=effects),ensure_ascii=False))
