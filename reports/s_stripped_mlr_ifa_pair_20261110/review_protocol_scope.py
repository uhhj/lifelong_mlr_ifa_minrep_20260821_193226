"""Read-only source/metric review; records reporting scope, never changes data."""
from pathlib import Path
import ast, datetime, hashlib, json

R = Path(__file__).resolve().parent
E = R.parent
REV = E / '20261110_reverse_order_main_protocol'
now = datetime.datetime.now().astimezone().isoformat()
J = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
H = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
paths = {
    'executed_protocol': R / 'PROTOCOL.md',
    'implementation_notes': R / 'IMPLEMENTATION_NOTES.md',
    'parent_audit': R / 'SOURCE_PARENT_AUDIT.md',
    'stripped_update': R / 'updates.py',
    'stripped_evaluator': R / 'cases/OBJECT_MLR_41/evaluate.py',
    'stripped_runtime': R / 'case_runtime_template.py',
    'main_evaluator': E / '20261013_coreset_object/source/coreset_exp/evaluate.py',
    'main_rollout': E / '20261013_coreset_object/source/coreset_exp/eval_metric.py',
    'main_metrics': E / '20261107_mlr_ifa_reproduction_audit/audit.py',
    'stripped_metrics': R / 'describe_final.py',
    'final_analysis': R / 'FINAL_ANALYSIS.json',
}
assert all(p.is_file() for p in paths.values())
def metric_ast(p):
    tree = ast.parse(p.read_text(encoding='utf-8-sig'))
    f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'metrics')
    ret = next(n for n in f.body if isinstance(n, ast.Return))
    assert isinstance(ret.value, ast.Call)
    return {x.arg: ast.dump(x.value, include_attributes=False) for x in ret.value.keywords}
old, new = metric_ast(paths['main_metrics']), metric_ast(paths['stripped_metrics'])
same = {k: old[k] == new[k] for k in ['AUC', 'FWT', 'NBT']}
assert all(same.values())
a = J(paths['final_analysis'])
metrics = {s: {m: a['summary'][s][m]['AUC'] for m in ['MLR', 'ALIGN', 'FULL_S', 'HISTORICAL_MLR']} for s in ['OBJECT', 'GOAL']}
holds = sorted(REV.glob('hold_dispatch_protocol_review_*/EVIDENCE.json'))
hold = J(holds[-1])
assert hold['status'] == 'DISPATCH_HELD_FOR_PROTOCOL_REVIEW_ACTIVE_CHILDREN_CONTINUE'
assert any(p['pid'] == 119466 and p['state'] == 'T' for p in hold['after'])
review = dict(
    status='EXECUTION_VERIFIED_PROTOCOL_SCOPE_RESTRICTED', observed=now,
    evidence={k: dict(path=str(p.relative_to(E.parent)), sha256=H(p)) for k,p in paths.items()},
    metric_formula_ast_equal=same, auc_percent=metrics,
    confirmed_changes=[
        'The new stripped MLR uses the full-S training lifecycle and a new BC+.1IFA update, rather than the historical PaperMLR.observe call chain.',
        'The new evaluation uses per-trajectory NumPy Normal streams rather than the historical global Torch Normal stream.',
        'The new evaluation caches single-frame spatial encodings rather than recomputing the full observation history.',
        'The new evaluation zeros actions after individual success and constructs environments per vector, unlike the historical evaluator.',
        'The S runtime disables cuDNN benchmark; the historical evaluation explicitly enabled it.',
    ],
    training_equivalence_limit='Direct-reference BC/IFA and recovery admissions are not a full original-observe equivalence proof.',
    authorization_limit='A self-authored frozen protocol and passed execution audit do not establish informed user approval for evaluator/training-entry changes.',
    timing_limit='Reviewed frozen files specify these choices before the new stripped runs. This review does not establish the absence of protocol changes in every earlier experiment.',
    interpretation='New MLR and ALIGN can be compared within their common stripped-S protocol. New MLR is not a drop-in historical main-table baseline; cross-protocol gains are not isolated mechanism effects.',
    result_policy='Preserve all measured values and raw evidence. Do not adjust RA or any other outcome to follow a changed baseline. No additional training/evaluation authorized by this review.',
    reverse_dispatch_hold_evidence=str(holds[-1].relative_to(E.parent)),
    new_optimizer_calls=0, new_environments=0, checkpoints_loaded=0,
)
(R/'PROTOCOL_SCOPE_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
note = '''# 协议与结果使用范围更正

本次审查确认：2026-11-10新建的剥离对照采用完整S训练骨架和评估入口。它不等同于主表历史MLR+IFA原生实现的复现。将此变化当作仅删除模块而未向用户明确说明并确认，属于协议处理错误。自写PROTOCOL、冻结hash及执行验收不能替代用户对科学协议变化的知情授权。

|项目|历史主表MLR+IFA|本轮剥离MLR/ALIGN|
|---|---|---|
|训练入口|原PaperMLR调用链|从S剥离的新BC+0.1IFA更新与阶段生命周期|
|动作噪声|全局Torch Normal随机流|逐轨迹固定NumPy Normal随机流|
|视觉历史计算|每步重新编码整段历史|单帧编码缓存后拼接|
|已成功轨迹动作|继续由策略生成|置零|
|环境生命周期|每个stage/task单元共用环境处理两组|每组新建并关闭环境|
|cuDNN benchmark评估设置|True|False|

原官方初始化、四阶段各100轮、BC+0.1IFA名义目标等保留项，不能证明完整训练调用链与原实现等价。本轮准入验证了新更新与显式公式及恢复的精确性，没有完成原observe完整调用链的等价验证。AUC/FWT/NBT的计算表达式AST逐项一致；问题在生成结果的训练与rollout流程，不能称为单纯改了表格或统计公式。

本轮新MLR与ALIGN使用共同实现和配对评估，可作为S实现下的剥离对照。它们不能直接替换历史主表MLR基线并与历史RA等结果解释为完全受控的模块增益。OBJECT的新MLR AUC为75.556%，历史为68.299%；GOAL分别75.833%与76.354%。目前没有分离各项改变的影响，不能把变化全部归给评估、训练入口或硬件中的任何一项。

所有实测值、检查点与原始证据保留。RA数值不得依据新基线人为调高；换基线不提供推算RA新成绩的依据。用户已说明额外时间与经济预算不可承担，本次更正不追加训练、闭环评估或补跑。论文可将新增对照单独报告，并明确历史结果的协议来源；不得将其混写成严格同协议对照。

当前逆序任务是另一组按各方法主表原生入口配置的运行，不与这轮剥离对照混称。2026-11-11已核对root/合同/PID后仅SIGSTOP派发器119466，暂停未来派发；两个在训子进程未收到停止信号。本地观察器32456已停止。该措施不表示科学运行最终完成，不改写退出码、不删除模型、不重训。

审查范围是上述已核实的剥离对照与派发状态，不能据此宣称其余历史实验已经全部通过协议一致性审查。原冻结协议与源文件保持原样；本文件纠正的是结果用途与对用户授权的解释。
'''
(R/'PROTOCOL_SCOPE_REVIEW.md').write_text(note,encoding='utf-8')
report=R/'REPORT.md'
body=report.read_text(encoding='utf-8')
banner='> 2026-11-11协议范围更正：本轮是S实现下的剥离对照，不能作为历史主表MLR+IFA的同协议替代。执行验收已完成不表示协议变化获用户授权。详见[PROTOCOL_SCOPE_REVIEW.md](PROTOCOL_SCOPE_REVIEW.md)。\n\n'
if banner not in body:
    first, rest=body.split('\n',1)
    report.write_text(first+'\n\n'+banner+rest.lstrip('\n'),encoding='utf-8')
state=J(R/'RUN_STATUS.json')
state['status']='COMPLETE_EXECUTION_VERIFIED_PROTOCOL_SCOPE_RESTRICTED'
state['protocol_review_observed']=now
state['protocol_scope_review']='PROTOCOL_SCOPE_REVIEW.json'
state['historical_main_baseline_replacement_permitted']=False
state['fabricated_or_adjusted_measurements']=False
(R/'RUN_STATUS.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

held=dict(status='PARTIAL_DISPATCH_HELD_PROTOCOL_REVIEW',observed=now,hold_evidence=str(holds[-1]),controller=119466,active_children_continue=True,local_observer_stopped=32456,scientific_source_changes=0,new_optimizer_calls=0,new_environments=0,note='No new dispatch or scientific rerun; original experimental outputs and contract retained.')
(REV/'PROTOCOL_REVIEW_HOLD.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name in ['RUN_STATUS.json','ACTIVE_ROOT.json']:
    state=J(REV/name)
    state['status']=held['status']
    state['protocol_review_hold']=held
    state['next_checkpoint_CST']=None
    (REV/name).write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
worker=J(REV/'CURRENT_TASK_WORKER.json')
worker.update(status='STOPPED_FOR_PROTOCOL_REVIEW',observed=now,next_checkpoint_CST=None)
(REV/'CURRENT_TASK_WORKER.json').write_text(json.dumps(worker,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name in ['REPORT.md','MONITOR_HANDOFF.md']:
    p=REV/name;body=p.read_text(encoding='utf-8')
    marker='> 协议审查暂停新增派发：controller119466已单独SIGSTOP；现有训练继续，本地观察器已停止，旧下一检查时间失效。不得自动CONT、重启或追加计算。详见PROTOCOL_REVIEW_HOLD.json及前序实验PROTOCOL_SCOPE_REVIEW.md。\n\n'
    if marker not in body:
        first,rest=body.split('\n',1);p.write_text(first+'\n\n'+marker+rest.lstrip('\n'),encoding='utf-8')
print(json.dumps(dict(status=review['status'],metric_formula_ast_equal=same,reverse_dispatch_held=True,new_optimizer=0,new_environments=0),ensure_ascii=False))
