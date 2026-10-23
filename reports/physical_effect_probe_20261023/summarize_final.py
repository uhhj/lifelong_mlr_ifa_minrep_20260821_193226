"""Build terminal Gate A report from independently accepted lightweight evidence."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil

R = Path(__file__).resolve().parent
S = R / 'snapshot_20261023_141307'
M = S / 'metadata'
F = R / 'final_A_20261023'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    audit = read(F / 'LOCAL_FINAL_AUDIT.json')
    snap = read(S / 'EVIDENCE.json')
    assert audit['status'] == 'PASS_LOCAL_COMPLETE_A'
    assert audit['scientific_decision'] == 'STOP_STATE_GATE'
    assert snap['controller_exit'] == '0' and not snap['controller_alive']
    assert not snap['hash_bad'] and not snap['gpu'].strip()
    results = read(M / 'STATE_PROBE_RESULTS.json')
    cost = read(M / 'COST_AUDIT.json')
    assert (cost['probe_fits'], cost['probe_epochs'], cost['probe_updates']) == (100, 5000, 41500)
    for name in ['INPUT_BINDING_AUDIT.json', 'NATIVE_VIEW_MANIFEST.json',
                 'VISIBILITY_AUDIT.json', 'DEMO_SPLITS.json',
                 'STATE_PROBE_RESULTS.json', 'A_GATE.json']:
        assert sha(M / name) == snap['files'][name]['sha256']
        shutil.copyfile(M / name, F / name)
    for name in ['PROTOCOL.md', 'USER_METHOD.txt']:
        shutil.copyfile(R / name, F / name)
    cost['independently_counted'] = audit['actual_cost']
    cost['additional_cpu_formal_window_forward_calls'] = 2
    cost['probe_pipeline_seconds'] = results['elapsed_seconds']
    cost['fit_seconds_sum'] = sum(x['seconds'] for x in cost['costs'])
    cost['status'] = 'COMPLETE_A_ONLY'
    write(F / 'COST_AUDIT.json', cost)
    for name in ['EFFECT_PREDICTION_RESULTS.json', 'GMM_INTERVENTION_RESULTS.json',
                 'ROLLOUT_STATE_ERRORS.json', 'CLOSED_LOOP_COMPARISON.json']:
        write(F / name, dict(status='NOT_RUN_A_GATE_FAILED',
              reason='State probe did not meet preregistered accuracy gate; B was not admitted',
              a_gate_sha256=sha(F / 'A_GATE.json'), policy_updates=0,
              formal_episodes=0, metrics=None))
    table = ['|任务|读出|物体－末端 median / P90 (mm)|物体－目标 median / P90 (mm)|',
             '|---|---|---:|---:|']
    for x in results['results']:
        if x['train_view'] == 'native' and x['test_view'] == 'native':
            v = x['interventions']['normal']
            a, b = v['object_eef']['all'], v['object_goal']['all']
            table.append(f"|{x['task']}|{x['model']}|{a['median_mm']:.3f} / {a['p90_mm']:.3f}|{b['median_mm']:.3f} / {b['p90_mm']:.3f}|")
    ci = []
    for c in audit['checks']['P3']['checks']:
        lo, hi = c['ci95_improvement_mm']
        ci.append(f"- Task{c['task']} / {c['target']}：[{lo:.3f}, {hi:.3f}] mm。")
    report = '''# 物理状态读出实验：A 关完整结果

状态：COMPLETED_GATED_STOP。A 关实验与独立验收完成，科学门槛未通过；按预注册条件结束本次流程，B 关及 seed42/43 未启动。这不是工程失败，也不是完整闭环负结果。

## 实际执行与验收

官方六任务预训练策略的视觉前端及 Temporal 冻结。OBJECT Task6/7 各 10 个 demo，以 demo 划分五折；每折 7 个训练、1 个验证、2 个测试。P0–P4、original/native 两种训练输入共 100 次拟合，每次 50 epoch；仅用验证集选 checkpoint。共 5,000 probe epoch、41,500 诊断优化步，策略更新 0、环境回合 0。训练输入/状态时序、无效边界 mask 与身份均已绑定原始 SHA。

服务器独立 CPU 审计核对原输入、100 个选中 checkpoint、30 组预测文件及全部指标；本地从 129,690 行误差标量独立重算 median/P90、demo 指标、动态子集及 bootstrap 门槛，LOCAL_FINAL_AUDIT 为 PASS_LOCAL_COMPLETE_A。冻结文件无变化，controller 正常 exit0、GPU 无残留。审计通过只证明执行和账目可信，不意味着科学门槛通过。

## 主要结果：native 训练 → native 测试

P0 为语言与本体状态历史；P1 为旧 Temporal 最新帧 token0；P2 为同一冻结 Temporal 的最新帧最后 token；P3 为完整模态同帧融合加因果时间读出；P4 将最新非语言 token 替换成前一帧。P0/P3/P4 使用匹配的轻量编码结构，P1/P2 使用匹配的回归头；不同结构的参数量详见成本表，不能称所有模型等算力。

''' + '\n'.join(table) + '''

## 为什么停止于 A

预注册要求两个任务、两个位置目标全部满足 median ≤10 mm、P90 ≤25 mm，并且相对 P0 的配对 demo-bootstrap 改善区间下界大于 0、动态当前状态优于前一状态且优于最新视觉替换。P2 与 P3 均未满足，selected=null。

P3 相比 P0 的 demo 误差改善 95% bootstrap 区间均为正：

''' + '\n'.join(ci) + '''

这支持“完整模态读出确实提取到有用的视觉状态信息”。但是 P3 的四组位置 median/P90 全部超出 10/25 mm 准入线，尤其物体－目标尾部误差仍为 46.84–59.73 mm；不能把相对改善当作控制精度达标。Task7 物体－末端的动态当前目标优势还为 −0.0455 mm。即使不考虑这个很小的动态差异，绝对精度条件仍明确失败。

仅把 P1 换成 P2 没有改善这些主要位置误差。可见性检查中，最新视觉/本体输入扰动对旧读出影响为 0，对完整最后 token 非零，说明旧读出存在结构限制；解除访问限制不等于冻结表示已能精确解码位置。

P3 与 P4 接近，Task7 物体－目标 P4 median 还略低（23.42 vs 24.41 mm），因此不能把全部收益归于“看见最新一帧”。但换成其他 demo 的视觉后，P3 的物体－目标 median 升至 Task6 约54.21、Task7 约48.06 mm，表明它确实使用了视觉条件，而非只复述本体历史。

## 输入域与局限

P3 在 original→original 与 original→native 的物体－末端 median，Task6 为11.70→14.94 mm、Task7 为11.52→13.44 mm，存在输入域影响；native→native 仍未达精度线，所以域偏差不是唯一解释。native 导出另与正式八帧观测路径做 CPU 校验，两个窗口最大差异分别约1.91e−5、1.28e−5，在预注册 CPU/GPU 容差内；未调用环境或策略优化。

结论只适用于当前冻结前端、预注册读出容量/预算和这两个任务的十 demo 分折。它不能证明表示在信息论上没有物体状态，也不能预测完整四任务 AUC。bootstrap 用于本次探索门槛，样本量小、交叉验证训练折重叠且比较多个模型，不作广义显著性或闭环收益声明。10/25 mm 是事前几何精度筛选，不是普适抓取成功保证。

## B、扩展与成本

没有训练效果预测器或 B0/B1/B2 控制策略，没有执行原计划的 300 个闭环回合，没有触发 seed42/43。相应结果文件明确写 NOT_RUN_A_GATE_FAILED，metrics=null，不能按零成功率计入比较。用户确认的扩展门槛保持不变：B2 两任务最终平均成功率严格高于两组对照，且最终旧 Task6 不低于任一对照，才整体扩展；当前未到该判定阶段。

100 个 probe 拟合耗时合计约461.86秒，probe 流程474.28秒；不包含实施、数据绑定、准入和独立验收。原生导出3,042帧另加640次复核，Temporal 特征18,252窗口、150批，诊断验证10,000批、选中测试1,350批；完整成本见 COST_AUDIT。所有模型、预测向量和缓存保留服务器，发布仅代码、元数据、误差标量和 hash。

前两个工程版本分别因身份歧义检查和 checkpoint 前缀适配在任何 probe 优化前停止；旧版本、原 exit1 和日志保留，第三版本使用冻结的新合同完成。另一次 CPU 路径检查因导入触发 EGL 初始化而失败，外置只读包装改用原函数 AST 后通过；没有修改科学源、重跑训练或环境。后续若尝试空间特征或新读出，需要新协议；本次没有依据结果调参或放宽门槛。

证据索引：同目录 LOCAL_FINAL_AUDIT.json / STATE_PROBE_RESULTS.json / A_GATE.json / COST_AUDIT.json；上级 snapshot_20261023_141307、independent_final_readonly_20261023_141335、error_scalars_readonly_20261023_141601、native_window_parity_readonly_20261023_141240。ARTIFACT_MANIFEST.json 绑定最终文件字节和 SHA。
'''
    (F / 'FINAL_REPORT.md').write_text(report, encoding='utf-8')
    (R / 'REPORT.md').write_text(report, encoding='utf-8')
    status = dict(status='COMPLETED_GATED_STOP', scientific_decision='STOP_STATE_GATE',
                  observation_time=datetime.datetime.now().astimezone().isoformat(),
                  A='COMPLETED_LOCAL_INDEPENDENT_PASS', B='NOT_RUN_A_GATE_FAILED',
                  seed_extension='NOT_TRIGGERED', final_directory=F.name,
                  controller_exit=0, controller_alive=False, gpu_idle_at_final_snapshot=True,
                  snapshot=S.name, policy_updates=0, formal_episodes=0,
                  local_audit_sha256=sha(F / 'LOCAL_FINAL_AUDIT.json'),
                  old_queues='CANCELLED_NOT_RESUMED')
    write(R / 'RUN_STATUS.json', status)
    (R / 'MONITOR_HANDOFF.md').write_text(
        '# Terminal handoff\n\nA completed with independent local PASS; scientific decision STOP_STATE_GATE. '
        'B and seed expansion were not admitted. Do not restart, retune, weaken gates or revive old queues. '
        'No monitor is needed for this terminal experiment. Preserve all source and failed-version evidence. '
        'Final report: final_A_20261023/FINAL_REPORT.md. Publication is verified separately by the latest '
        'PUBLICATION_RECEIPT; never infer push success from this status.\n', encoding='utf-8')
    write(F / 'ARTIFACT_MANIFEST.json', {p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
          for p in sorted(F.iterdir()) if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json'})
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
