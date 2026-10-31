"""Recompute historical baseline and latest gated S metrics from accepted matrices."""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import statistics as st

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
SOURCES = {
    'GOAL': 'execution/20261029_support_gate_full_s_goal/final_readonly_20261030_111546',
    'OBJECT': 'execution/20261028_support_gate_full_s_object/final_readonly_20261029_201235',
}

def metrics(m):
    v = {k: F(str(x))/100 for k,x in m.items()}
    avg = lambda a: sum(a)/len(a)
    return {
        'AUC': avg([avg([v[f'{s}_{t}'] for s in range(t,10)]) for t in range(6,10)]),
        'FWT': avg([v[f'{t}_{t}'] for t in range(6,10)]),
        'NBT': avg([avg([v[f'{t}_{t}']-v[f'{s}_{t}'] for s in range(t+1,10)]) for t in range(6,9)]),
        'OldFinal': avg([v[f'9_{t}'] for t in range(6,9)]),
        'FinalAll': avg([v[f'9_{t}'] for t in range(6,10)]),
    }

def main():
    bindings, results = {}, {}
    def read(rel):
        raw = (WORK/rel).read_bytes()
        bindings[rel] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return json.loads(raw)
    for suite, directory in SOURCES.items():
        base = read(f'execution/20261025_ccbp_v4_full/{suite}_THREE_SEED_BASELINES.json')
        latest = read(directory+'/LOCAL_FINAL_AUDIT.json')
        summary = read(directory+'/RESULT_SUMMARY.json')
        assert summary['final_audit_sha256'] == bindings[directory+'/LOCAL_FINAL_AUDIT.json']['sha256']
        rows=[]
        for method in ['MLR_IFA', 'LATEST_S']:
            for seed in [41,42,43]:
                r = next(r for r in base['runs'] if r['seed']==seed and r['method']=='MLR_IFA') if method=='MLR_IFA' else latest['seeds'][str(seed)]
                matrix = r['matrix'] if method=='MLR_IFA' else {k:v['percent'] for k,v in r['matrix'].items()}
                computed = metrics(matrix)
                for k,v in computed.items():
                    assert v == F(r['metrics'][k]['fraction'])
                    assert abs(float(v*100)-r['metrics'][k]['percent']) < 1e-9
                rows.append(dict(method=method, seed=seed, matrix=matrix, metrics={k:float(v*100) for k,v in computed.items()}))
        stats={}
        for method in ['MLR_IFA','LATEST_S']:
            stats[method]={k:dict(mean=st.mean(r['metrics'][k] for r in rows if r['method']==method), sample_sd=st.stdev(r['metrics'][k] for r in rows if r['method']==method)) for k in computed}
        for k in computed:
            for stat in ['mean','sample_sd']:
                assert abs(stats['LATEST_S'][k][stat]-summary['statistics'][k][stat]) < 1e-9
        results[suite]=dict(per_seed=rows, summary=stats, delta_pp={k:stats['LATEST_S'][k]['mean']-stats['MLR_IFA'][k]['mean'] for k in computed})
    audit=dict(status='PASS_MATRIX_METRIC_RECOMPUTATION', suites=results, source_bindings=bindings, new_training=0, new_environment=0,
               limits=['Historical baseline versus latest support-weighted gated S; not original ungated SHBR.', 'No verified common action noise with historical baseline; unequal compute and training interactions.', 'OBJECT seed43 baseline is frozen repeat1; no result-based replacement.', 'Three development seeds; sample SD is not a confidence interval.'])
    (HERE/'COMPARISON.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 原MLR＋IFA与最新S指标比较','', 'GOAL与OBJECT，各seed41/42/43；均值±样本标准差，单位%。差值为S减基线，单位百分点。最新S为RA＋ALIGN、专家纠错及成功历史保持，并包含支持度加权和候选更新门控；不是早期未门控SHBR。', '', '|套件|指标|原MLR＋IFA|最新S|差值pp|','|---|---|---:|---:|---:|']
    for suite,r in results.items():
        for k,d in r['delta_pp'].items():
            a,b=(r['summary'][m][k] for m in ['MLR_IFA','LATEST_S'])
            lines.append(f"|{suite}|{k}|{a['mean']:.2f} ± {a['sample_sd']:.2f}|{b['mean']:.2f} ± {b['sample_sd']:.2f}|{d:+.2f}|")
    lines += ['', 'AUC、FWT、OldFinal和FinalAll越高越好；NBT越低越好。AUC按任务先平均后续阶段、再跨任务平均；FWT沿项目定义为任务获得时成功率均值，不是零样本前向迁移。OldFinal为最终旧三任务均值，FinalAll为最终四任务均值。', '', '最新S六个seed/套件组合的AUC均高于原基线。GOAL整体和最终旧任务水平更高，但NBT更高，不能解释为遗忘全面改善；NBT以各自获得时表现为参照，与最终绝对成功率不同。', '', '比较使用项目复现结果，不是论文原表。历史基线没有与最新S核实共同动作噪声；S有额外采集及优化器成本，不能将差异单独归因于SHBR或称等预算/统计显著。OBJECT基线seed43沿冻结协议使用repeat1；若换为首次运行，历史基线AUC均值为70.5903%，S差值为+15.9028pp，作为敏感性说明而非替换主口径。', '', '逐seed矩阵、精确指标复算和来源SHA见COMPARISON.json。本轮没有训练、仿真或远程实验检查。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:16]))

if __name__=='__main__':
    main()
