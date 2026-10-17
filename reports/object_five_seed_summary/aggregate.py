"""Read-only recomputation: first runs 40-44; seed43 repeat kept separate."""
import hashlib
import json
import statistics
from fractions import Fraction as F
from pathlib import Path

OUT = Path(__file__).resolve().parent
EX = OUT/'evidence'
METHODS = ('random', 'coreset', 'event_coreset')
LABELS = dict(zip(METHODS, ('Random', 'Coreset', 'Event+Coreset')))
KEYS = ('AUC', 'FWT', 'NBT')
hashes = {}

def read(p):
    data = p.read_bytes()
    hashes[p.relative_to(OUT).as_posix()] = hashlib.sha256(data).hexdigest()
    return data.decode('utf-8-sig')

def js(p):
    return json.loads(read(p))

new = EX/'20261016_seeds40_41_three_arm/completion_evidence'
rc = EX/'20261013_coreset_extension/final_evidence/remote'
ec = {
    42: EX/'20261015_event_coreset/final_evidence/run',
    43: EX/'20261016_seed43_readonly_audit/completed_evidence/EC',
    44: OUT/'sources/EC44',
}
repeat = EX/'20261016_seed43_three_arm_repeat_v1/completion_evidence'
accepted_new = js(new/'queue/FINAL_ACCEPTANCE.json')
accepted_repeat = js(repeat/'queue/FINAL_ACCEPTANCE.json')
assert accepted_new['status'] == accepted_repeat['status'] == 'COMPLETED'
remote_hashes = js(ec[44]/'REMOTE_SHA256.json')
for name, expected in remote_hashes.items():
    assert hashlib.sha256((ec[44]/name).read_bytes()).hexdigest() == expected, name

def compute(seed, method, folder, expected, run='first'):
    rates, sources = {}, []
    for stage in range(6, 10):
        for task in range(6, stage+1):
            p = folder/f'trials_stage{stage}_task{task}.jsonl'
            records = [json.loads(line) for line in read(p).splitlines() if line.strip()]
            assert len(records) == 20, p
            assert {r['trial'] for r in records} == set(range(20)), p
            assert {r['initial_state_index'] for r in records} == set(range(20)), p
            for r in records:
                assert r['stage'] == stage and r['task'] == task and r['epoch'] == 99, p
                assert r['seed'] == seed and r['arm'] == method, p
                assert type(r['success']) is bool, p
            rates[stage,task] = F(sum(r['success'] for r in records),20)
            sources.append(p.relative_to(OUT).as_posix())
    metrics = {
        'AUC': sum(sum(rates[s,t] for s in range(t,10))/ (10-t) for t in range(6,10))/4,
        'FWT': sum(rates[t,t] for t in range(6,10))/4,
        'NBT': sum(sum(rates[t,t]-rates[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
    }
    for k,v in metrics.items():
        e = expected[k]
        if isinstance(e,dict):
            assert v == F(e['fraction']), (seed,method,run,k,v,e)
            e = e['percent']
        assert abs(float(v*100)-e)<1e-10, (seed,method,run,k,v,e)
    return dict(seed=seed,method=method,run=run,episodes=200,
                metrics={k:dict(fraction=str(v),percent=float(v*100)) for k,v in metrics.items()},
                matrix=[[float(rates[s,t]*100) if t<=s else None for t in range(6,10)] for s in range(6,10)],
                sources=sources)

first = []
for seed in range(40,45):
    for method in METHODS:
        if seed in (40,41):
            folder=new/f'seed{seed}'/('ec' if method=='event_coreset' else 'rc')/method/'evaluation'
            expected=accepted_new['results'][str(seed)][method]['metrics']
        elif method=='event_coreset':
            folder=ec[seed]/method/'evaluation'
            expected=js(ec[seed]/'SUMMARY.json')['metrics']
        else:
            folder=rc/f'seed{seed}'/method/'evaluation'
            expected=js(rc/f'seed{seed}'/'SUMMARY.json')['arms'][method]['metrics']
        first.append(compute(seed,method,folder,expected))

reruns=[]
for method,short in zip(METHODS,('R','C','EC')):
    folder=repeat/('EC' if short=='EC' else 'RC')/method/'evaluation'
    reruns.append(compute(43,method,folder,accepted_repeat['results'][short]['repeat1'],'repeat1'))

stats={m:{k:{'mean_percent':statistics.mean(r['metrics'][k]['percent'] for r in first if r['method']==m),
             'sample_sd_percent':statistics.stdev(r['metrics'][k]['percent'] for r in first if r['method']==m)}
          for k in KEYS} for m in METHODS}
paired={}
for a,b in [('coreset','random'),('event_coreset','coreset'),('event_coreset','random')]:
    vals=[next(r for r in first if r['seed']==s and r['method']==a)['metrics']['AUC']['percent']-
          next(r for r in first if r['seed']==s and r['method']==b)['metrics']['AUC']['percent'] for s in range(40,45)]
    paired[a+' minus '+b]={'auc_deltas_pp':vals,'mean_pp':statistics.mean(vals),'sample_sd_pp':statistics.stdev(vals)}
assert all(hashlib.sha256((OUT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
result=dict(audit='PASS',first_run_episodes=3000,repeat43_episodes=600,sd_ddof=1,
            seed43_repeat_excluded=True,source_hashes_unchanged=True,
            first_runs=first,seed43_repeat=reruns,five_seed_statistics=stats,paired_auc_deltas=paired)
(OUT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'SOURCE_SHA256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')

lines=['# LIBERO-OBJECT 五 seed 汇总','',
       '三种方法均开启 IFA、use_state=True；只统计原协议四阶段 ep99 的 200 条闭环/组。',
       'seed40–44 每个 seed 使用首次结果。seed43 repeat1 单独列出，不替换首次结果，不纳入五 seed 均值与标准差。',
       '数值为百分制；AUC/FWT 越高越好，NBT 越低越好。标准差为跨 seed 样本标准差（ddof=1）。','',
       '| Seed | 方法 | AUC ↑ | FWT ↑ | NBT ↓ |','|---|---|---:|---:|---:|']
for r in first:
    lines.append('| '+str(r['seed'])+' | '+LABELS[r['method']]+' | '+' | '.join(f"{r['metrics'][k]['percent']:.3f}" for k in KEYS)+' |')
for m in METHODS:
    lines.append('| 五 seed 均值±SD | '+LABELS[m]+' | '+' | '.join(f"{stats[m][k]['mean_percent']:.3f} ± {stats[m][k]['sample_sd_percent']:.3f}" for k in KEYS)+' |')
lines+=['','## seed43 完整重跑（不计入上述统计）','','| 方法 | AUC ↑ | FWT ↑ | NBT ↓ |','|---|---:|---:|---:|']
for r in reruns:
    lines.append('| '+LABELS[r['method']]+' | '+' | '.join(f"{r['metrics'][k]['percent']:.3f}" for k in KEYS)+' |')
lines+=['','## 复算与边界','',
        '从 180 个逐 episode 文件复算：首次 3,000 条，seed43 重跑 600 条；每单元 20 个唯一 trial/初态，阶段、任务、epoch99、seed、arm、二值结果校验通过；全部指标与已有汇总一致。',
        '本次补齐 EC44 远端原始轻量记录，其 SHA256 与远端一致；全部本地输入分析前后 SHA256 不变。未训练、未追加评估、未加载权重。',
        'SOURCE_SHA256.json 保存输入来源与哈希，RESULTS.json 保存完整精度、全部矩阵及配对差值。',
        '五 seed 均值仅描述这些探索性运行。seed43 重跑显示结果波动，且原评估未隔离逐轨迹动作随机流；不能仅凭均值或标准差宣称统计显著或普遍因果。',
        'NBT 受刚学完时的基准成功率影响，不能单凭更负的 NBT 判定绝对任务表现更好。','',
        '复现命令：`python -X utf8 reports/object_five_seed_summary/aggregate.py`。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(audit='PASS',stats=stats,paired_auc=paired,source_files=len(hashes)),ensure_ascii=False,indent=2))
