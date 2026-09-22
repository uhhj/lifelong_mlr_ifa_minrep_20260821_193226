import ast,inspect,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .io import write_json
from qp_lbfgsb_kkt_driven_v1.offline import OUT,mp
from sample_qp_lifelong_v1_continuation_v3 import common as c
from sample_qp_optimized_v1_continuation_v4 import solver,core,trust
from sample_qp_lifelong_v1_continuation_v4 import step,train

REPORT=c.ROOT/'reports/sample_qp_lifelong_v1_continuation_v4'

def main():
    hard=c.read(OUT/'HARD_QP_REGRESSION_V3.json');normal=c.read(OUT/'NORMAL_NONZERO_GPU_PARITY.json')
    b258=c.read(OUT/'writeback/B258/WRITEBACK.json');b185=c.read(OUT/'writeback/B185/WRITEBACK.json')
    unit=(REPORT/'UNIT_TESTS.log').read_text();assert 'Ran 16 tests' in unit and unit.rstrip().endswith('OK')
    assert core.solve is solver.solve and step.project is core.project
    identities={m.__name__:dict(path=inspect.getsourcefile(m),sha256=c.sha(inspect.getsourcefile(m))) for m in (solver,core,trust,step,train)}
    call_checks={}
    for mod,callee in [(train,'update'),(step,'project'),(core,'solve')]:
        tree=ast.parse(Path(inspect.getsourcefile(mod)).read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==callee]
        call_checks[mod.__name__]=bool(calls) and all({'diag_dir','context'}.issubset({k.arg for k in n.keywords}) for n in calls)
    assert all(call_checks.values())
    write_json(REPORT/'RUNTIME_BINDING.json',dict(status='PASS',identity=True,call_checks=call_checks,modules=identities,sitecustomize_required=False,formal_training_resumed=False))
    # Forced rejection smoke of the exact v4 solver. Synthetic return is clearly labeled.
    z=np.load(OUT/'frozen_b258/before_fallback.npz');smoke=REPORT/'failure_dump_smoke'
    def fail(A,b,x):
        bad=np.zeros_like(x);r=solver.parts(A,b,bad)
        return bad,bad,[dict(iteration=1,**r)],dict(nit=1,accepted=False,message='SMOKE synthetic forced trust failure',selected=r),bad[None,:]
    ctx=dict(synthetic_forced_trust_failure=True,task=8,epoch=11,batch=258,pre_global_step=42057,optimizer_step_before=3657,scheduler_step_before=3657,record_indices=c.read(OUT/'frozen_b258/FAILURE_BATCH_INPUT.json')['record_indices'])
    raised=False
    with patch.object(solver,'trust_run',side_effect=fail):
        try:solver.solve(z['K'],z['f'],float(z['u_norm']),diag_dir=smoke,context=ctx)
        except AssertionError as e:
            assert 'failed frozen KKT after both' in str(e);raised=True
    saved=np.load(smoke/'QP_RESULT.npz');meta=c.read(smoke/'QP_RESULT.json')
    checks=dict(raised_after_dump=raised,raw_arrays_present={'K','f','A','b','x_standard','lbfgsb_candidate','trust_candidate','lbfgsb_iterates','trust_iterates'}.issubset(saved.files),context_exact=meta['context']==ctx,not_accepted=not meta['accepted'],both_trajectories=bool(meta['lbfgsb_trace']) and bool(meta['trust_trace']))
    assert all(checks.values());write_json(REPORT/'FAILURE_DUMP_SMOKE.json',dict(status='PASS',checks=checks,source='actual v4 solver with synthetic forced trust rejection',formal_training_resumed=False))
    # Package transport uses identical numeric implementation, verified again after all tests.
    cand=c.ROOT/'qp_trust_continuation_candidate_v1';opt=c.ROOT/'sample_qp_optimized_v1_continuation_v4';tr=c.ROOT/'sample_qp_lifelong_v1_continuation_v4'
    eq={name:(opt/name).read_bytes()==(cand/name).read_bytes() for name in ['solver.py','core.py','trust.py','numeric.py','io.py']}
    eq['step.py']=(tr/'step.py').read_text().replace('from sample_qp_optimized_v1_continuation_v4.core import','from .core import')==(cand/'step.py').read_text();assert all(eq.values())
    # Independent KKT recheck of all accepted solutions; no numerical solve here.
    from decimal import Decimal
    precision={}
    for name,r in hard['results'].items():
        sol=np.load(OUT/'hard'/name/'accepted.npz');p=sol['b']-sol['A']@sol['x'];k=max(np.maximum(p,0).max(),np.maximum(-sol['x'],0).max(),np.abs(sol['x']*p).max());hp=mp(sol['A'],sol['b'],sol['x']);precision[name]=bool(k<=1e-8 and Decimal(hp['kkt'])<=Decimal('1e-8'))
    assert all(precision.values())
    formal_files=c.read(REPORT/'PROTOCOL_FROZEN.json')['files'];scoped={str(p.relative_to(c.ROOT)):c.sha(p)==formal_files[str(p.relative_to(c.ROOT))] for d in [opt,tr] for p in d.glob('*.py')};assert all(scoped.values())
    parent=c.read(c.OUT/'PROTOCOL_FROZEN.json')['files'];old={k:c.sha(c.ROOT/k)==parent[k] for k in ['sample_qp_optimized_v1_continuation_v3/solver.py','sample_qp_optimized_v1_continuation_v3/core.py','sample_qp_lifelong_v1_continuation_v3/step.py','sample_qp_lifelong_v1_continuation_v3/train.py']};assert all(old.values())
    gates=dict(unit_tests=True,F1=hard['results']['F1']['status']=='PASS',F2=hard['results']['F2']['status']=='PASS',B185=hard['results']['B185']['status']=='PASS',B258=hard['results']['B258']['status']=='PASS',decimal80_all=all(precision.values()),B258_writeback=b258['status']=='PASS',B185_writeback=b185['status']=='PASS',normal_32_64_96=normal['status']=='PASS',failure_dump=all(checks.values()),checkpoint_restore=all(b258['reload'].values()) and all(b185['reload'].values()),runtime_binding=all(call_checks.values()),candidate_code_transfer=all(eq.values()))
    assert all(gates.values())
    report=dict(status='PASS',V4_PREFLIGHT_PASS=True,gates=gates,unit_tests=16,hard_results=hard['results'],B258_writeback=dict(physical_residual=b258['record']['projection_residual'],actual_positive=b258['record']['actual_positive'],optimizer=[b258['optimizer_before'],b258['optimizer_after']],scheduler=[b258['scheduler_before'],b258['scheduler_after']],post_state=b258['post_state']),B185_writeback=dict(physical_residual=b185['record']['projection_residual'],historical_state_exact=b185['checks']['historical_v3_final_state_exact']),normal_nonzero_results=normal['results'],numeric_code_transfer=eq,scoped_source_integrity=scoped,parent_sources_unchanged=old,frozen_protocol_sha256=c.sha(REPORT/'PROTOCOL_FROZEN.json'),evidence_directory=str(OUT),recovery_scope='Diagnostic states restore model/AdamW/scheduler/RNG exactly; no formal canonical recovery slot migrated or training launched in this round.',full_repository_hash_performed=False,formal_training_resumed=False)
    write_json(REPORT/'PREFLIGHT_COMPLETE.json',report)
    write_json(REPORT/'STATUS.json',dict(status='PREFLIGHT_PASS_TRAINING_NOT_STARTED',V4_PREFLIGHT_PASS=True,formal_training_resumed=False))
    write_json(OUT/'STATUS.json',dict(stage='V4_PREFLIGHT_PASS_TRAINING_NOT_STARTED',formal_training_resumed=False))
    text='''# batch258 等价第二求解器验证\n\n正式训练未恢复。QP、normalization、lambda映射、1e-8门限及AdamW语义均未修改。\n\n## 数值结论\n\n原v3 L-BFGS-B在1次迭代后以目标相对变化停止，KKT=1.547045713756745e-8。指定的ftol=gtol=0候选运行39次仍失败，最小callback KKT=1.1223141533529099e-8，final=2.694640956587779e-8；该失败候选未纳入v4。\n\ntrust-constr按固定配置从同一30k标准解出发，第40次首次通过：float64=3.3966850332638125e-9，80位十进制=3.3966852472411387e-9。最终第50次以gtol停止，final KKT=1.944270892788285e-11；按照预定规则接受第40次，而不是事后挑更优最终解。\n\n## 门禁\n\n16项单元测试通过；F1/F2/B185/B258均通过两种精度验收。F1不调用fallback，F2/B185沿用原v3第一fallback，仅B258调用trust-constr。\n\nB258真实单步的physical residual=0，actual_positive=0，optimizer与scheduler均3657→3658；模型/校正/位移均finite。B185结果与历史v3完整状态精确相同，physical residual=3.7383576701034376e-10。两项post-step状态保存后restore全部一致。每个hard batch只执行一次native更新，没有前序batch重放。\n\n32/64/96普通路径在非零LR、非零位移上均与v3的模型、moments、scheduler、RNG逐项完全相同，两套fallback均未调用。保留warmup首步零位移检查，但以补充的非零位移结果作为充分证据。\n\n正式v4来自数值代码逐字节相同的候选包，step仅改import路径；已验证函数identity、train/step/core的显式捕获参数传递，并在真实v4 solver中强制失败验证数组/轨迹/上下文先落盘再raise。\n\n## 资产审计更正\n\n旧qp_solver_recovery_v2/F2_before_extended.npz已被历史流程覆写为B185内容，首次使用它的回归作废并保留在invalid_F2_source_attempt。最终F2取v3冻结fixtures/F2_dual_qp.npz，并核验其原始残差1.0038755857838533e-8，不能将两个B185误当四个独立实例。\n\n## 当前边界\n\nV4_PREFLIGHT_PASS=true。未启动train或评估，也未生成正式RECOVERY_INDEX。保存的post-step文件是可直接恢复的诊断状态，不冒充完整epoch恢复点。后续正式启动仍需使用既有parent-child恢复机制迁移合法边界。\n\n正式v3源码和旧资产保持不变；未做全仓hash。四个hard实例的通过不保证所有未来QP都会成功，最终失败仍保留完整现场并fail-stop。\n'''
    (REPORT/'FINAL_REPORT.md').write_text(text)
    write_json(OUT/'FINAL_SUMMARY.json',dict(status='V4_PREFLIGHT_PASS',report=str(REPORT/'FINAL_REPORT.md'),formal_training_resumed=False))
    print('V4_PREFLIGHT_PASS true; formal_training_resumed false',flush=True)

if __name__=='__main__':main()
