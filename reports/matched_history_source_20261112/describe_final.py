"""Aggregate accepted per-case audits and write the paired comparison."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import sys


ROOT = Path(__file__).resolve().parent
SUITES = ('GOAL','OBJECT')
SEEDS = (41,42,43)


def main(evidence):
    receipt = json.loads(evidence.read_bytes())
    assert receipt['exit'] == 0
    audit = receipt['remote']
    completion = audit['completion']
    cases = audit['cases']
    assert completion['controller_exit'] == 0 and completion['residual_experiment_processes'] == 0
    assert completion['frozen_files'] == 233
    assert set(cases) == {'%s_%d'%(suite,seed) for suite in SUITES for seed in SEEDS}
    assert audit['controller']['phase'] == 'complete_pending_final_audit'
    assert sum(case['formal_episodes'] for case in cases.values()) == 960
    assert sum(case['collection_episodes'] for case in cases.values()) == 180
    assert sum(case['initial_state_bytes_checked'] for case in cases.values()) == 720
    assert sum(x['accepted_updates'] for v in cases.values() for x in v['training'].values()) == 419400
    assert sum(x['production_optimizer_calls'] for v in cases.values() for x in v['training'].values()) == 838800
    results = {}
    for suite in SUITES:
        seeds = {}
        for seed in SEEDS:
            case = cases['%s_%d'%(suite,seed)]
            groups = {}
            for arm in 'ES':
                counts = case['counts'][arm]
                assert set(counts)=={'6','7','8','9'} and all(0<=n<=20 for n in counts.values())
                groups[arm] = {**{'Task'+t:n*5 for t,n in counts.items()},
                               'OldFinal':sum(counts[t] for t in '678')*100/60,
                               'FinalAll':sum(counts.values())*100/80,
                               'GateQ':100*case['training'][arm]['gate_Q_accepts']/case['training'][arm]['gate_blocks']}
            for t, pair in case['paired'].items():
                assert pair['S_only']-pair['E_only']==case['counts']['S'][t]-case['counts']['E'][t]
            seeds[str(seed)]={**groups,'S_minus_E':{k:groups['S'][k]-groups['E'][k] for k in groups['E']},
                              'paired_discordant':case['paired']}
        avg={}
        for arm in ('E','S','S_minus_E'):
            avg[arm]={k:{'mean':mean(v[arm][k] for v in seeds.values()),
                         'sample_sd':stdev(v[arm][k] for v in seeds.values())}
                      for k in seeds['41'][arm]}
        results[suite]={'seeds':seeds,'three_seed':avg}
    payload = dict(status='COMPLETE_ACCEPTED', completion=completion, formal_episodes=960,
                   training_collection_episodes=180, total_physical_training_collection_episodes=190,
                   accepted_updates=419400, production_optimizer_calls=838800,
                   admission_optimizer_calls=348, results=results, audit_evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
    (ROOT/'FINAL_RESULTS.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    columns=('Task6','Task7','Task8','Task9','OldFinal','FinalAll','GateQ')
    time=datetime.fromtimestamp(max(c['evaluation_completed_epoch_s'] for c in cases.values()),
                                timezone(timedelta(hours=8))).isoformat()
    report=['# 专家示范历史与自主成功历史的配对闭环验收','',
            '最终闭环完成时间（北京时间） '+time+'。六对各完成两个 Task9 分支，每个分支 100 轮和 80 个闭环回合。',
            '共 419,400 次接受更新、838,800 次生产优化器调用和 960 个正式回合。E 为专家示范历史，S 为自主成功历史。指标单位为百分比，差值单位为百分点。','']
    for suite, table in results.items():
        report+=['## LIBERO-'+suite,'','| Seed | 组 | '+' | '.join(columns)+' |',
                 '|---|---|'+'---:|'*len(columns)]
        for seed,row in table['seeds'].items():
            for arm in 'ES':
                report+=['| '+seed+' | '+arm+' | '+' | '.join('%.2f'%row[arm][k] for k in columns)+' |']
        for arm in ('E','S','S_minus_E'):
            ds=table['three_seed'][arm]
            report+=['| 三种子 | '+arm+' | '+' | '.join('%.2f ± %.2f'%(ds[k]['mean'],ds[k]['sample_sd']) for k in columns)+' |']
        report+=['','| Seed | Task6 S独胜/E独胜 | Task7 S独胜/E独胜 | Task8 S独胜/E独胜 | Task9 S独胜/E独胜 |',
                 '|---|---:|---:|---:|---:|']
        for seed,row in table['seeds'].items():
            report+=['| '+seed+' | '+' | '.join('%d/%d'%(row['paired_discordant'][t]['S_only'],
                row['paired_discordant'][t]['E_only']) for t in '6789')+' |']
        report+=['']
    report+=['## 配对结论','',
             'GOAL 的旧任务平均成功率提高 4.44 个百分点，三个种子分别提高 10.00、0.00 和 3.33 个百分点，主要增益来自 Task6。Task9 的三种子均值相同，全任务平均成功率提高 3.33 个百分点。',
             'OBJECT 的旧任务平均成功率下降 6.11 个百分点，三个种子分别下降 5.00、3.33 和 10.00 个百分点，主要下降来自 Task6。Task8 平均提高 3.33 个百分点，全任务平均成功率下降 5.42 个百分点。',
             '这组对照支持自主成功历史在 GOAL 上带来保持收益。OBJECT 上的方向相反，跨套件的一致优势未得到支持。','',
             '## 验收范围','',
             '核对冻结文件、12 个最终检查点的严格 CPU 加载和恢复字段、18 个教师来源与缓存、候选门控和训练预算，以及 960 个评估回合与 180 个训练采集回合的初态字节和噪声哈希。教师重放使用缓存中的输入，不重演物理环境。',
             '原 GOAL seed42 Task7 采集因窗口计数配置错误在写缓存前退出。接续版保留已有结果及 Task6 缓存，经授权重采 10 回合。原记录留档，重采回合成功与否及终止步数 10/10 一致，动作序列哈希 8/10 一致。实际训练采集共执行 190 回合。',
             '本消融在同一 Task8 起点上比较 Task9 阶段的保持梯度输入来源。两个组的门控均访问共同探针。三个训练种子的配对均值与样本标准差描述本次结果，完整多阶段 DEBR 的效果另由主实验检验。']
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main(ROOT / sys.argv[1])
