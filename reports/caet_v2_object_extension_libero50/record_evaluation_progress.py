"""Record an audited local snapshot without server mutation or metric peeking."""
import datetime
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
p = Path(sys.argv[1]).resolve()
d = json.loads((p / 'SNAPSHOT.json').read_text(encoding='utf8'))
a = json.loads((p / 'LOCAL_AUDIT.json').read_text(encoding='utf8'))
e = json.loads((p / 'EVALUATION_LOCAL_AUDIT.json').read_text(encoding='utf8'))
assert a['epochs'] == 400 and a['main_optimizer_steps'] == 105500
assert a['frozen_hashes'] == 507 and not d['partial_aux']
out = '/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed41_20261021'
for task in range(6, 10):
    for phase in ('train', 'retain'):
        assert d['files'][f'{out}/{phase}{task}.exit'].strip() == '0'
assert a['task6_parity']
assert any('coreset_exp.evaluate' in x['command'] for x in d['processes'])
ev = e['formal']
assert ev['technical_exceptions'] == 0
note = '''最新2026-10-22 01:31 CST：progress_seed41_20261022_013003完成400epoch105500主步，train6–9/retain6–9全部exit0，507冻结hash与Task6实际tensor同RA、本地逐epoch/逐步账目PASS。Task8/9迁移实际7602/9103步，冲突关闭6224/6998，保护非零2915/4186步；两预测器留出PASS，拟合额外1840更新。Task7全RA回退。正式60/200条、6完成vector+1进行中、0技术异常；evaluate PID72624及10环境正常。新增外置audit_evaluation_progress.py <目录> <seed>，逐episode初态hash/index/envseed/600步及attempt唯一性/成功向量独立PASS，cell summary留最终包。6批均249.01秒，剩余约58分钟加最终审计，预计02:30–03:00 CST供参考。仍PARTIAL，无部分AUC或性能结论。未重启、未改科学源、未重复预检；43和LIBERO50仍排队未GPU准入。容量watcher仍ARMED未接管，data普通空闲9.205GB/root全空闲14.075GB，system普通5.902GB；无需对活动payload操作。下一轮继续核41正式评估，完成后package_final/audit_final 41并确认43自动准入。'''
report = f'''# CAET-v2 OBJECT seed41 评估进度

状态：PARTIAL；观察时间 {a['observed_at']}。

全部400epoch、105500主更新已完成，各阶段训练与入库退出0。507项冻结源码/输入hash通过；Task6实际tensor与父RA一致。最终checkpoint严格加载及全流程退出尚待验收。

|阶段|迁移应用步|冲突关闭步|保护非零步|
|---|---:|---:|---:|
|Task8|7602|6224|2915|
|Task9|9103|6998|4186|

两效果预测器留出均通过；额外效果拟合1840更新独立计数。Task7零支持，整阶段回退RA。Task8拟合墙钟负耗时原字段仍保留且标不可用，不参与可靠耗时合计。这些是机制执行证据，不是性能收益。

正式评估{ev['records']}/200条，{ev['completed_vectors']}个vector完成、{ev['started_vectors']-ev['completed_vectors']}个在运行，0技术异常。evaluate PID72624与10个原生环境运行。逐episode初态身份、env seed、600步上限、attempt唯一性和成功向量已本地独立复算；本快照未取cell summary，最终包再核验。未计算部分指标。

已完成6批平均{ev['mean_vector_seconds']:.2f}秒；按此估计余评估约{ev['remaining_evaluation_hours']*60:.1f}分钟，另计最终审计，预计02:30–03:00 CST完成seed41，仅供参考。seed43与LIBERO50仍排队且尚未GPU准入。容量后备保持ARMED，没有接管或修改当前实验。

本轮仅只读检查与本地审计，无训练/评估追加、重启或冻结源码修改。后续须200条/20vector、4CP严格加载、全部有效exit0及无残留，再做本地最终验收。全队列尚未完成。
'''
(p / 'REPORT.md').write_text(report, encoding='utf8')
for name in ('MONITOR_HANDOFF.md', 'REPORT.md'):
    with (root / name).open('a', encoding='utf8') as f:
        f.write('\n\n## 2026-10-22 01:31更新\n\n' + note + '\n')
s = json.loads((root / 'RUN_STATUS.json').read_text(encoding='utf8'))
s.update(observed_at=a['observed_at'], epochs=400, steps=105500,
         formal_records=60, completed_vectors=6, active_phase='EVALUATING',
         latest_progress=p.name, latest_queue='queue_20261022_012942',
         eta_object41='2026-10-22 02:30–03:00 CST (measured reference)',
         auxiliary=a['auxiliary'])
(root / 'RUN_STATUS.json').write_text(json.dumps(s, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
(p / 'HANDOFF_UPDATE.txt').write_text(note, encoding='utf8')
print(note)
