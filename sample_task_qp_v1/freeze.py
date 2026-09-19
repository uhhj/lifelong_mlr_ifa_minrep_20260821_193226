import time,shutil
from .common import *
def run():
    OUT.mkdir(exist_ok=False)
    old=read(OLD/'PROTOCOL_FROZEN.json');limits=dict(old['resources'])
    limits.update(main_measurement_per_origin_seconds_excluding_recovery=1800,main_queue_wall_hours_including_recovery_and_IO=12,new_data_disk_artifacts_max_GiB=2,new_system_scratch_max_GiB=2)
    p=dict(name='sample_task_qp_v1',frozen_at=time.time(),points=old['points'],preflight6=old['preflight6'],seeds=old['seeds'],arms=ARMS,horizons=[1],
      purpose='Describe how per-window versus task-mean constraints change a single update. No rollout, no multi-step continuation, no carrier regeneration.',
      primary_pair=['TASK_GRAD_QP','SAMPLE_GRAD_QP'],secondary_mechanistic_pair=['TASK_STEP_QP','SAMPLE_STEP_QP'],
      primary_reason='Match latest task-gradient QP implementation; actual-step pair separately isolates AdamW transformation.',
      sampling='Exact existing32 fixed P windows per old task; no new sampling or reweighting. A sample is a temporal window, not an independent demonstration. Preserve demo lineage and report grouped effects.',
      constraints='For every valid old row h, require h^T v<=0 in the projected descent/displacement variable. Task rows are original32-window mean gradients; sample rows are individual full GMM NLL gradients from the same windows. Include all valid rows; no initially-violated-only prefilter.',
      task_baseline='Reuse original task gradient evaluator and exact small active-set solver; require original first-step task-arm state witnesses exactly, plus gradient mean decomposition tolerance.',
      sample_solver='Full-space cone projection via normalized small Gram dual SLSQP with exact coordinate-minimization KKT refinement; no Fisher/rank truncation, no ridge, no slack, no kappa. Independent full-column scipy.nnls at all six engineering origins.',
      intervention='Gradient pair modifies mixed BC gradient before original clipping/AdamW. Step pair modifies native AdamW parameter displacement and retains native moments. Same parameter scope, original mixed BC weights, RNG and scheduler.',
      isolation='Freeze all five candidate updates and decision hashes before opening A/B. P is training-accessible. Reuse original A/B whole20 trajectory-isolated development monitors; no claim of historically unseen data.',
      metrics=['raw projected norm and rotation','actual parameter norm and rotation vs MLR','requested and realized correction','near-stall ratio<=0.1 descriptive marker','positive first-order rates for every old window before and after AdamW/parameter rounding','task-average versus sample masking counts','finite full GMM NLL on every P and A/B window','demo/group and task/seed means, harm rates and tails','new task NLL cost','optimizer moments divergence','time and memory'],
      numeric=dict(reference_norm_squared_min=1e-12,dual_KKT_normalized=1e-8,projected_float32_rate_max=5e-6,sample_to_task_gradient_relative=2e-5,sample_to_task_gradient_absolute=1e-6,full_NNLS_correction_relative=2e-6,finite_loss_floor='64*float32eps*max(1,abs(initial),abs(MLR))'),
      guarantees='QP feasibility is first-order on the measured rows. Actual AdamW movement, finite NLL and isolated generalization are separately measured, never inferred from solver success. Zero is feasible only for this unrestricted homogeneous cone.',
      decisions='Descriptive one-step mechanism report only. No controller effectiveness, persistence, or closed-loop claim. All36 points retained; zero interventions and failed geometry estimates are not filtered.',
      resources=limits,gpu_wait_hours=24,no_hardware_or_carrier_substitution=True)
    write(OUT/'PROTOCOL_FROZEN.json',p)
    shutil.copyfile(OLD/'PROBE_WINDOW_MEMBERSHIP.json',OUT/'PROBE_WINDOW_MEMBERSHIP.json')
    files=[OUT/'PROTOCOL_FROZEN.json',OUT/'PROBE_WINDOW_MEMBERSHIP.json',OLD/'PROTOCOL_FROZEN.json',OLD/'ASSET_AUDIT.json',ROOT/'reports/sensitive_component_compatibility_v1/PROBE_MEMBERSHIP.json',ROOT/'reports/gradient_relation_4090_v2/DESIGN.json.gz']
    files += [ROOT/'reports/group_joint_update_v1/measurements'/point/'SELECTION.json' for point in old['points']]
    for pkg in ['sample_task_qp_v1','paper_active_set_v1','functional_geometry_diagnostic_v1','gradient_relation_experiment','trajectory_relation_diagnostic_v1','sensitive_component_compatibility_v1','cb_ifa_experiment']:
        files+=list((ROOT/pkg).glob('*.py'))
    write(OUT/'FROZEN_INPUTS.json',{x.relative_to(ROOT).as_posix():sha(x) for x in files})
    (OUT/'PROTOCOL.md').write_text('''# 逐样本与任务平均QP：单步更新诊断

复用原36个3090起点、seed45/46/47及每个旧任务固定32个buffer窗口。每个起点五个候选均来自同一参数、真实AdamW、scheduler、RNG与混合batch。

主比较：TASK_GRAD_QP vs SAMPLE_GRAD_QP。辅助成对比较：TASK_STEP_QP vs SAMPLE_STEP_QP。MLR为共同参照。两种位置分别比较，不能把梯度层/位移层差异误认成样本粒度效应。

本轮只测第1步，不跑闭环，不跑20步续训，不重建载体，不增加Fisher、软系数、容忍损伤或新任务预算。样本指原来的时间窗口，保留demo来源；32窗口不等于32条独立演示。

同时记录：更新缩小与旋转、修正量、所有样本的一阶残差、任务平均掩盖的样本冲突、实际写入后的残差、探测集与隔离A/B的完整GMM NLL、逐任务/逐demo/逐seed损益和新任务代价。先冻结候选，后读取A/B。

先完成CPU资产审计与求解器测试；GPU恢复后须通过原状态与原下一步逐位核验、六点工程检查，才完成36点。失败停止，不换样本、不换载体，不声称当前约束保证实际BC或成功率。
''',encoding='utf-8')
    print('PROTOCOL_FROZEN',sha(OUT/'PROTOCOL_FROZEN.json'))
if __name__=='__main__':run()
