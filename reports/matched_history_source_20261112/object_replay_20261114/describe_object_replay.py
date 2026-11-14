"""Compare the audited replay with the archived OBJECT evaluation."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys


ROOT = Path(__file__).resolve().parent


def main(name):
    audit = ROOT / name
    receipt = json.loads(audit.read_bytes())
    assert receipt['exit'] == 0
    result = receipt['remote']
    assert result['status'] == 'COMPLETE_ACCEPTED' and result['formal_episodes'] == 480
    original = json.loads((ROOT/'FINAL_RESULTS.json').read_bytes())['results']['OBJECT']
    result['audit_sha256'] = hashlib.sha256(audit.read_bytes()).hexdigest()
    results = result['results']
    assert set(results) == {'OBJECT_41','OBJECT_42','OBJECT_43'}
    finished = datetime.fromtimestamp(max(v['evaluation_completed_epoch_s'] for v in results.values()),
                                      timezone(timedelta(hours=8))).isoformat()
    report=['# OBJECT 闭环复评','',
            '复评完成时间（北京时间） '+finished+'。',
            '同一批六个 Task9 检查点，沿用初态 0–19、相同噪声与原评估实现。三个种子各评估 160 回合，共 480 回合。',
            'E 为专家示范历史组，S 为自主成功历史组。下表单位为百分比，箭头前后分别为原评与复评。','',
            '| Seed | 组 | Task6 | Task7 | Task8 | Task9 | OldFinal | FinalAll |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    metrics=('Task6','Task7','Task8','Task9','OldFinal','FinalAll')
    for case, item in sorted(results.items()):
        seed=case.rsplit('_',1)[1]
        item['metrics']={}
        for arm in 'ES':
            counts=item['counts'][arm]
            assert len(counts)==4 and all(0<=v<=20 for v in counts.values())
            new={**{'Task'+k:v*5 for k,v in counts.items()},
                 'OldFinal':sum(counts[k] for k in '678')*100/60,
                 'FinalAll':sum(counts.values())*100/80}
            item['metrics'][arm]=new
            old=original['seeds'][seed][arm]
            report+=['| '+seed+' | '+arm+' | '+' | '.join('%.2f → %.2f'%(old[k],new[k]) for k in metrics)+' |']
    report+=['','| 三种子均值 | 原评 E | 复评 E | 原评 S | 复评 S |',
             '|---|---:|---:|---:|---:|']
    ag={}
    for k in metrics:
        nums=[v for arm in 'ES' for v in (original['three_seed'][arm][k]['mean'],
                                        mean(x['metrics'][arm][k] for x in results.values()))]
        ag[k]={'original_E':nums[0],'replay_E':nums[1],'original_S':nums[2],'replay_S':nums[3]}
        report+=['| '+k+' | '+' | '.join('%.2f'%v for v in nums)+' |']
    result['comparison_means']=ag
    result['identical_per_episode']={k:sum(v[k] for v in results.values()) for k in
                                    ('success_matches','step_matches','action_matches')}
    m=result['identical_per_episode']
    report+=['','逐回合与原评匹配的数量为成败 %d/480、终止步数 %d/480、动作序列哈希 %d/480。'
             %(m['success_matches'],m['step_matches'],m['action_matches']),
             '复评使用相同检查点、初态和噪声，衡量闭环重复性。独立训练种子仍为三个。']
    (ROOT/'OBJECT_REPLAY_FINAL_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (ROOT/'OBJECT_REPLAY_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')


if __name__=='__main__':
    main(sys.argv[1])
