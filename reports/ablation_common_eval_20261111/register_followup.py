"""Validate the actual CPU preparation before the continuation is armed."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

from audit_results import reference_cases, summarize
from common import ROOT, read, sha
from transport import FILES


def main(evidence):
    spec = read(ROOT / 'SPEC.json')
    result = read(evidence)
    assert result['status'] == 'READY_AWAITING_PREDECESSOR' and result['formal_episodes'] == result['new_optimizer'] == 0
    contract = dict(files={name: sha(ROOT / name) for name in FILES}, case_count=12, episode_count=2400)
    assert hashlib.sha256((json.dumps(contract, indent=2) + '\n').encode()).hexdigest() == result['contract']
    assert set(result['cpu']) == set(spec['evaluation_cases'])
    for name, planned in spec['evaluation_cases'].items():
        cpu = result['cpu'][name]
        assert cpu['case'] == name and cpu['status'] == 'PASS_CPU_PREPARATION' and cpu['initial_bytes_verified'] == 80
        assert cpu['new_optimizer'] == cpu['new_environment'] == 0
        assert set(cpu['checkpoints']) == set('6789')
        for stage, actual in cpu['checkpoints'].items():
            assert actual == dict(sha256=planned['models'][stage]['sha256'], keys=100, parameters=21633611)
    historical_path = ROOT.parent / '20261111_main_ablation_protocol_audit/AUDIT.json'
    assert sha(historical_path) == spec['table_audit_sha256']
    historical = read(historical_path)['cases']
    for pair, source in spec['comparison_outcomes'].items():
        suite, seed = pair.split('_')
        for method, rows in reference_cases(source).items():
            expected = historical[f'{method}/{suite}/{seed}']['metrics_percent']
            actual = summarize(rows)['metrics_percent']
            assert all(abs(actual[key] - expected[key]) < 1e-9 for key in actual)
    observed = datetime.datetime.fromtimestamp(result['observed'], datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    registration = dict(status='READY_AWAITING_REVERSE_ACCEPTANCE', observed=observed, predecessor=spec['predecessor'],
        server_root='/data/mlr_ifa/experiment_workspaces/' + spec['name'], contract=result['contract'],
        preparation_evidence=evidence.relative_to(ROOT).as_posix(), preparation_sha256=sha(evidence),
        runs=12, verified_checkpoints=48, formal_planned=2400, formal_started=0, new_optimizer=0,
        retained_reference_runs=18, reference_metrics_recomputed=True)
    with (ROOT / 'QUEUE_REGISTRATION.json').open('x', encoding='utf-8') as file:
        json.dump(registration, file, indent=2)
        file.write('\n')
    report = f'''# 论文消融表统一复评

PARTIAL。{observed} 完成 CPU 预检。复评接在当前逆序实验自然完成并验收之后。

MLR+IFA 和 RA-Coreset 在 LIBERO-OBJECT 与 LIBERO-GOAL 各复评 seed41、42、43。沿用论文正序 Task6→7→8→9 和 48 份已保存阶段模型。每个运行共 200 回合，每个阶段与已学任务组合使用原初态 0–19，共 2,400 回合。C、Q 与完整 DEBR 使用原先已验收的 18 个运行结果。

复评使用服务器上原完整 DEBR 评估器的相同字节，采用长度 8 历史缓存、逐回合固定 Normal 噪声、两组各 10 个初态、600 步上限和原成功判定。仅适配模型路径和新输出位置。CPU 核对了 48 个模型的历史 SHA、严格载入以及原初态文件内容。GPU 缓存验证随每个正式评估运行进行。

第一版 CPU 准备在正式回合前发现 Windows 换行与服务器源码字节不符，保留原错误记录和服务器目录。第二版使用原始 LF 字节并通过预检。排队时新增训练和正式回合均为 0。
'''
    (ROOT / 'REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps(registration))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
