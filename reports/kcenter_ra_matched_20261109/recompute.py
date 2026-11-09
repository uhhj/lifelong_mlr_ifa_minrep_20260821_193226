"""Read-only recomputation of the K-Center parents actually bound by RA."""
from pathlib import Path
from fractions import Fraction as F
import csv
import datetime
import hashlib
import json
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {}

def raw(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    b = p.read_bytes()
    SOURCES[p.relative_to(ROOT).as_posix()] = hashlib.sha256(b).hexdigest()
    return b

def read(path):
    return json.loads(raw(path).decode('utf-8-sig'))

def metrics(m):
    return dict(AUC=sum(sum(m[s,t] for s in range(t,10))/ (10-t) for t in range(6,10))/4,
        FWT=sum(m[t,t] for t in range(6,10))/4,
        NBT=sum(sum(m[t,t]-m[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
        OldFinal=sum(m[9,t] for t in range(6,9))/3,
        NewFinal=m[9,9], FinalAll=sum(m[9,t] for t in range(6,10))/4)

def packed(vals):
    return {k:dict(fraction=str(v),percent=float(v*100)) for k,v in vals.items()}

obj = read('execution/20261019_ra_object_conditional/BASELINE_EPISODE_AUDIT.json')
five = read('execution/20261017_five_seed_summary/RESULTS.json')
g41 = read('execution/20261019_ra_coreset_goal_seed41/progress_20261019_222601/completion_baseline/SUMMARY.json')
gext = read('execution/20261017_coreset_goal_three_seed/FINAL_RECOMPUTED.json')
protocol_files = [
 '20261019_ra_object_conditional/PROTOCOL.md',
 '20261019_ra_object_conditional/PARENTS.json',
 '20261019_ra_object_conditional/completion_20261020/LOCAL_FINAL_ACCEPTANCE.json',
 '20261019_ra_coreset_goal_seed41/PROTOCOL.md',
 '20261019_ra_coreset_goal_seed41/progress_20261019_222601/LOCAL_FINAL_ACCEPTANCE.json',
 '20261018_ra_coreset_goal_seed42/PROTOCOL.md',
 '20261018_ra_coreset_goal_seed42/LOCAL_FINAL_ACCEPTANCE.json',
 '20261019_ra_coreset_goal_extension/EXTENSION_PROTOCOL.md',
 '20261019_ra_coreset_goal_extension/progress_20261019_160822/LOCAL_FINAL_ACCEPTANCE.json']
for p in protocol_files:
    raw('execution/'+p)

runs = []
identities = {}
for suite in ['OBJECT','GOAL']:
    baseline = read(f'execution/20261025_ccbp_v4_full/{suite}_THREE_SEED_BASELINES.json')
    baseline_hashes = {str(Path(p).resolve()):h for p,h in baseline['source_sha256'].items()}
    for seed in [41,42,43]:
        ra_ref = next(x for x in baseline['runs'] if x['method']=='RA' and x['seed']==seed)
        ra_files = [Path(p) for p in baseline['source_sha256']
                    if 'ra_coreset' in p and 'baseline' in Path(p).parts and 'trials_stage' in p and
                    (f'final_{suite}_{seed}_' in p or (suite=='OBJECT' and seed==41 and 'final_ALIGN_seed41_' in p))]
        assert len(ra_files)==10, (suite,seed,len(ra_files))
        if suite=='OBJECT':
            candidates = five['seed43_repeat'] if seed==43 else five['first_runs']
            ref = next(x for x in candidates if x['seed']==seed and x['method']=='coreset')
            kfiles = [Path(p) for p in ref['sources']]
            expected = obj['seeds'][str(seed)]['metrics']['coreset']
            k_hashes = {Path(p).name:h for p,h in obj['seeds'][str(seed)]['input_episode_sha256'].items() if '/coreset/' in p}
        else:
            kdir = (ROOT/'execution/20261019_ra_coreset_goal_seed41/progress_20261019_222601/completion_baseline/coreset/evaluation') if seed==41 else (ROOT/f'execution/20261017_coreset_goal_three_seed/final_evidence/seed{seed}/coreset/evaluation')
            kfiles = sorted(kdir.glob('trials_stage*_task*.jsonl'))
            expected = (g41['arms']['coreset'] if seed==41 else gext['seeds'][str(seed)]['coreset'])['metrics']
            k_hashes = None
        for method,files,reference in [('K-Center',kfiles,expected),('RA',ra_files,ra_ref['metrics'])]:
            assert len(files)==10
            m={}; ids={}; paths=[]
            for p in files:
                b=raw(p); digest=hashlib.sha256(b).hexdigest()
                if method=='RA': assert digest==baseline_hashes[str(p.resolve())]
                elif k_hashes: assert digest==k_hashes[p.name]
                rows=[json.loads(line) for line in b.decode('utf-8-sig').splitlines() if line.strip()]
                assert len(rows)==20
                s,t=rows[0]['stage'],rows[0]['task']
                assert (s,t) not in m
                assert sorted(r['initial_state_index'] for r in rows)==list(range(20))
                assert all(r['stage']==s and r['task']==t and r['seed']==seed and isinstance(r['success'],bool) for r in rows)
                m[s,t]=F(sum(r['success'] for r in rows),20)
                ids[s,t]={r['initial_state_index']:r['initial_state_sha256'] for r in rows}
                paths.append(p.relative_to(ROOT).as_posix())
            assert set(m)=={(s,t) for s in range(6,10) for t in range(6,s+1)}
            vals=metrics(m)
            for k,v in reference.items(): assert vals[k]==F(v['fraction']), (suite,seed,method,k)
            identities[suite,seed,method]=ids
            runs.append(dict(suite=suite,seed=seed,method=method,records=200,
                variant='repeat1' if suite=='OBJECT' and seed==43 and method=='K-Center' else 'historically_bound',
                metrics=packed(vals), matrix={f'{s}_{t}':float(v*100) for (s,t),v in sorted(m.items())}, sources=sorted(paths)))
        assert identities[suite,seed,'RA']==identities[suite,seed,'K-Center'], (suite,seed,'initial states')

names=list(runs[0]['metrics'])
summary={}; deltas=[]
for suite in ['OBJECT','GOAL']:
    summary[suite]={}
    for method in ['K-Center','RA']:
        rr=[r for r in runs if r['suite']==suite and r['method']==method]
        summary[suite][method]={k:dict(mean_percent=float(sum(F(r['metrics'][k]['fraction']) for r in rr)/3*100),sample_sd_pp=statistics.stdev(r['metrics'][k]['percent'] for r in rr)) for k in names}
    for seed in [41,42,43]:
        a=next(r for r in runs if (r['suite'],r['seed'],r['method'])==(suite,seed,'RA'))
        b=next(r for r in runs if (r['suite'],r['seed'],r['method'])==(suite,seed,'K-Center'))
        vals={k:F(a['metrics'][k]['fraction'])-F(b['metrics'][k]['fraction']) for k in names}
        deltas.append(dict(suite=suite,seed=seed,metrics=packed(vals),cell_delta_pp={c:a['matrix'][c]-b['matrix'][c] for c in a['matrix']}))
    dd=[d for d in deltas if d['suite']==suite]
    summary[suite]['RA_minus_KCenter']={k:dict(mean_pp=float(sum(F(d['metrics'][k]['fraction']) for d in dd)/3*100),sample_sd_pp=statistics.stdev(d['metrics'][k]['percent'] for d in dd)) for k in names}

old43=next(r for r in five['first_runs'] if r['seed']==43 and r['method']=='coreset')
out=dict(status='PASS_READONLY_MATCHED_PROTOCOL_RECOMPUTATION', observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    seeds=[41,42,43], records_recomputed=2400, initial_state_identity_pairs_checked=1200,
    new_training=0,new_forward=0,new_environment=0,runs=runs,summary=summary,per_seed_deltas=deltas,
    object43_parent='repeat1, fixed in the historical RA protocol before this audit; original first-run excluded from main matched comparison',
    object43_original_unmatched_metrics=old43['metrics'],
    scope='MLR+IFA with pure K-Center versus RA selection; no ALIGN, execution retention, support calibration or gate.',
    definitions=dict(AUC='mean over tasks of mean success over stages since introduction', FWT='mean diagonal success',NBT='mean over old tasks of mean diagonal-minus-later success; signed, lower is better',OldFinal='mean final success of tasks6-8',NewFinal='final task9 success',FinalAll='mean final success of tasks6-9',sd='sample SD across 3 seeds, ddof=1'),
    limits=['Historical protocol-matched comparison, not a new hardware-controlled rerun.', 'OBJECT parents41/43 used RTX4090 whereas corresponding RA used RTX3090; retain historical hardware boundary.', 'Evaluation initial states match, but historical stage-global action RNG does not guarantee identical per-episode action noise.', 'Three seeds; report mixed directions and do not infer statistical significance.', 'Original OBJECT43 first run remains retained; do not substitute it for the pre-bound repeat1 parent.'],
    source_sha256=SOURCES)
for p,h in SOURCES.items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
(HERE/'RESULTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
with (HERE/'METRICS.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.writer(f);w.writerow(['suite','seed','method',*names])
    for r in runs:w.writerow([r['suite'],r['seed'],r['method'],*[r['metrics'][k]['percent'] for k in names]])
    for r in deltas:w.writerow([r['suite'],r['seed'],'RA_minus_KCenter_pp',*[r['metrics'][k]['percent'] for k in names]])
lines=['# K-Center 与纯 RA：seed41/42/43 同协议历史结果复算','',
 '两个套件均找到三个 seed 的完整 K-Center 和纯 RA 结果。本次复算 12 个 run、120 个单元、2400 条正式回合记录，逐项核对 1200 对初始状态身份，全部已有指标与接受记录的精确分数一致；未新增训练、前向或环境调用。','',
 '比较对象均保留 MLR+IFA，RA 仅改变记忆选样（随机锚点保护与覆盖交换），此处不含 ALIGN、执行保持、支持度校准或门控。每 run Task6–9 各100轮、每单元20回合、每 run 200正式回合，原600步/5zero评估。OBJECT 与 GOAL 各自保留原训练配置，不跨套件混算。','',
 '**版本选择：OBJECT seed43 用 repeat1 K-Center**，这是 RA 当时 PROTOCOL/PARENTS/BASELINE_EPISODE_AUDIT 冻结的实际父基线，不是本次按结果选取。原 seed43 首次版本仍保留在 RESULTS.json，不纳入主比较。GOAL 使用各自原 K-Center41/42/43；不混入历史42/43/44均值。','',
 '下表单位为百分数，差为 RA−K-Center 的百分点；NBT 越低越好，其余越高越好。三 seed 标准差为样本SD（ddof=1）。','']
for suite in ['OBJECT','GOAL']:
    lines += [f'## {suite}', '', '|方法|AUC|FWT|NBT↓|OldFinal|NewFinal|FinalAll|','|---|---:|---:|---:|---:|---:|---:|']
    for method in ['K-Center','RA']:
        lines.append('|'+method+'|'+'|'.join(f"{summary[suite][method][k]['mean_percent']:.4f} ± {summary[suite][method][k]['sample_sd_pp']:.4f}" for k in names)+'|')
    lines.append('|RA−K-Center|'+'|'.join(f"{summary[suite]['RA_minus_KCenter'][k]['mean_pp']:+.4f}" for k in names)+'|')
    lines += ['', '|seed|方法|AUC|FWT|NBT↓|OldFinal|NewFinal|FinalAll|','|---|---|---:|---:|---:|---:|---:|---:|']
    for seed in [41,42,43]:
        for method in ['K-Center','RA']:
            r=next(x for x in runs if (x['suite'],x['seed'],x['method'])==(suite,seed,method))
            lines.append(f'|{seed}|{method}|'+'|'.join(f"{r['metrics'][k]['percent']:.4f}" for k in names)+'|')
        r=next(x for x in deltas if (x['suite'],x['seed'])==(suite,seed))
        lines.append(f'|{seed}|差值|'+'|'.join(f"{r['metrics'][k]['percent']:+.4f}" for k in names)+'|')
    lines += ['', '### 逐阶段/任务单元差值（RA−K-Center，pp）','', '|seed|'+ '|'.join(runs[0]['matrix'])+'|','|---|'+ '|'.join(['---:']*10)+'|']
    for d in deltas:
        if d['suite']==suite:lines.append(f"|{d['seed']}|"+'|'.join(f'{v:+.1f}' for v in d['cell_delta_pp'].values())+'|')
    lines.append('')
lines += ['## 结果解释','',
 'GOAL：RA 的 AUC 与最终成功率在三个 seed 上都高于 K-Center，NBT 在两个 seed 上更低、一个持平；这是本队列中更一致的选样收益。',
 'OBJECT：RA 的平均 AUC 提高0.9375pp，但逐 seed 为两升一降；最终成功率一升两降，均值下降0.8333pp，最终旧任务成功率下降2.2222pp。K-Center 的 NBT 更低，不能表述为 RA 在所有指标上更好或普遍改善遗忘。NBT 以各自刚学完的分数为参照，需结合较高的 RA FWT 与最终旧任务成功率一起看。','',
 '可支持的结论：相同历史训练/评估协议下，RA 相对纯 K-Center 的优势主要体现在 GOAL（AUC+5.6250pp，最终成功率+8.3333pp）；OBJECT 的 AUC/FWT 较高，但最终表现与NBT未改善，且存在种子依赖性。这组结果支持覆盖与随机锚点组合具有实用价值，但不足以证明每个任务套件、每个种子或每项指标均获益。','',
 '## 可比性边界与定义','',
 '上述“同协议”指各 seed 对应的冻结数据、初始化、预算、任务顺序、训练目标及评估规则；不是完全同硬件重新对照。OBJECT seed41/43 的 K-Center 父运行在4090，RA在3090。历史接受记录核验了Task6 tensor/初始RNG等一致性，但闭环结果仍存在执行差异。正式初态身份一致不等于逐轨迹行动噪声完全一致（原评估按stage重置全局RNG）。不将全部观测差异作单机制严格因果解释，不以三seed均值声称统计显著。','',
 '令 r(s,t) 为 stage s 在 task t 的成功率。AUC = mean_t mean_{s≥t} r(s,t)；FWT = mean_t r(t,t)；NBT = mean_{t=6..8} mean_{s>t}[r(t,t)−r(s,t)]，保留负值；OldFinal = mean_{t=6..8} r(9,t)；NewFinal = r(9,9)；FinalAll = mean_{t=6..9} r(9,t)。','',
 '完整逐seed指标见 METRICS.csv；各run的10单元矩阵、精确分数、原始回合路径、源SHA、父基线绑定见 RESULTS.json。recompute.py 可完全离线复算。']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],records=2400,summary=summary),ensure_ascii=False,indent=2))
