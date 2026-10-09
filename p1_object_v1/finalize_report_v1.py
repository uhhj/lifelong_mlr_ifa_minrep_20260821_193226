"""Post-run reporting repair; never trains, evaluates, or alters frozen runtime.

The original queue passed r twice to dict(). The legacy metric helper also
rounds display values. This reader uses the same formulas with exact rational
arithmetic for the investment gate and keeps legacy rounded values separately.
"""
import csv
import json
import time
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path

from .io import write,sha
from .prepare import ROOT,OUT
from .legacy_metrics import metrics,fixed_epoch_matrix


def exact_metrics(matrix):
    r=[[Fraction(str(x)) if x is not None else None for x in row] for row in matrix]
    n=len(r)
    fwt=sum(r[j][j] for j in range(n))/n
    nbt=sum(sum(r[j][j]-r[i][j] for i in range(j+1,n))/(n-j-1) for j in range(n-1))/(n-1)
    auc=sum(sum(r[i][j] for i in range(j,n))/(n-j) for j in range(n))/n
    return dict(auc=auc,fwt=fwt,nbt=nbt)


def floats(values):return {k:float(v) for k,v in values.items()}


def summarize(seed=42):
    import torch
    report=OUT/('seed%d'%seed);run=ROOT/('runs/p1_object_v1/seed%d'%seed)
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    checks={}
    checks['frozen_runtime_unchanged']=all(sha(ROOT/f)==h for f,h in protocol['implementation_files'].items())
    checks['training_complete']=json.loads((run/'TRAINING_COMPLETE.json').read_text())['status']=='PASS'
    checks['evaluation_sentinel']=(run/'PRIMARY_EVAL_COMPLETE').is_file()
    with (report/'training_log.csv').open() as f:epochs=list(csv.DictReader(f))
    checks['400_unique_epochs']=len(epochs)==400 and {(int(r['stage']),int(r['epoch'])) for r in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
    rows=[json.loads(s) for s in (report/'trials_primary.jsonl').read_text().splitlines()]
    groups=defaultdict(list)
    for row in rows:groups[(row['stage'],row['epoch'],row['task'])].append(row)
    expected={(s,99,t) for s in range(6,10) for t in range(6,s+1)}
    checks['200_trials_10_cells']=len(rows)==200 and set(groups)==expected
    checks['20_unique_original_initial_states_per_cell']=all(len(rs)==20 and {r['initial_state_index'] for r in rs}==set(range(20)) for rs in groups.values())
    matrix=[[None]*4 for _ in range(4)]
    for (s,e,t),rs in groups.items():matrix[s-6][t-6]=100*sum(r['success'] for r in rs)/len(rs)
    logged,_=fixed_epoch_matrix(run/'primary_ep99.log')
    checks['trial_matrix_matches_native_log']=matrix==logged
    primary=exact_metrics(matrix);baseline_matrix=protocol['baseline_results'][str(seed)]['r'];base=exact_metrics(baseline_matrix)
    checks['legacy_formula_display_parity']=all(round(float(primary[k]),3)==metrics(matrix)[k] for k in primary)
    checkpoints={t:run/'checkpoints'/('mlrifa_%d_model_ep99.pth'%t) for t in range(6,10)}
    checkpoint_hashes={t:sha(p) for t,p in checkpoints.items()}
    checks['trial_checkpoint_identity']=all(Path(r['checkpoint'])==checkpoints[r['stage']] for r in rows)
    prior={};teachers=[];active_by_stage={}
    for task in range(6,10):
        active=json.loads((run/'teacher_targets'/('ACTIVE_task%d.json'%task)).read_text())['windows']
        assert all(active[k]==v for k,v in prior.items() if k in active),'OLD_TEACHER_REFRESHED'
        table_path=run/'teacher_targets'/('task%d_ep99.pt'%task)
        table=torch.load(table_path,map_location='cpu')
        assert table['seed']==seed and table['source_task']==task and table['epoch']==99
        assert table['checkpoint_sha256']==checkpoint_hashes[task]
        assert all(torch.isfinite(table[k]).all() for k in ['log_weights','means','scales']) and (table['scales']>0).all()
        assert table['means'].shape==(len(table['ids']),5,7) and table['scales'].shape==table['means'].shape
        assert table['log_weights'].shape==(len(table['ids']),5)
        assert (run/'transactions'/('task%d'%task)/'COMMITTED.json').is_file()
        teachers.append(dict(task=task,new_count=len(table['ids']),active_count=len(active),table_sha256=sha(table_path),checkpoint_sha256=checkpoint_hashes[task]))
        active_by_stage[task]=active;prior=active
    checks['four_immutable_own_ep99_teacher_transactions']=True
    recovery=torch.load(run/'recovery/latest.pt',map_location='cpu')
    checks['legal_final_epoch_recovery']=recovery['task']==9 and recovery['next_epoch']==100 and recovery['stage_complete'] and recovery['state']['optimizer_step']==recovery['state']['scheduler_step']
    from .teacher_store import TeacherStore
    store=TeacherStore(run/'teacher_targets',seed,recovery['teacher_manifest'])
    checks['final_buffer_teacher_bijection']=store.validate(recovery['buffer_dir'])
    stages={};events_manifest=[]
    for task in range(6,10):
        stats=Counter();sums=defaultdict(float)
        allowed={} if task==6 else active_by_stage[task-1]
        for epoch in range(100):
            path=report/'events'/('task%d_epoch%03d.jsonl'%(task,epoch));events_manifest.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),size=path.stat().st_size))
            with path.open() as f:
                for line in f:
                    row=json.loads(line);assert row['task']==task and row['epoch']==epoch and row['seed']==seed
                    stats['steps']+=1;assert row['step']==stats['steps']
                    ids=row.get('old_ids',[]);assert len(ids)==row['old_count'] and len(set(ids))==len(ids)
                    assert all(i in allowed for i in ids)
                    assert row.get('teacher_stages',[])==[allowed[i]['source_task'] for i in ids]
                    if ids:assert row['teacher_hit_rate']==1.
                    else:assert row['policy_kl']==0. and row['policy_weighted']==0.
                    assert abs(row['policy_weighted']-.05*row['policy_kl'])<1e-10
                    stats['distill_steps']+=bool(ids);stats['old_windows']+=len(ids)
                    stats['negative_kl_windows']+=row.get('negative_kl_windows',0)
                    stats['clipped_steps']+=row['gradient_clipped']
                    for k in ['bc','ifa_weighted','policy_kl','policy_weighted','total_loss','distill_forward_seconds']:sums[k]+=row[k]
        stages[str(task)]=dict(stats,mean_per_step={k:v/stats['steps'] for k,v in sums.items() if k!='distill_forward_seconds'},distill_forward_seconds=sums['distill_forward_seconds'])
    training=json.loads((run/'TRAINING_COMPLETE.json').read_text())
    checks['event_counts_match_training_recovery']=all(sum(v[k] for v in stages.values())==training['totals'][k]==recovery['totals'][k] for k in ['steps','distill_steps','old_windows','negative_kl_windows'])
    checks['no_secondary_trials']=not (report/'trials_secondary.jsonl').exists()
    assert all(checks.values()),checks
    additional=primary['auc']>base['auc']
    result=dict(seed=seed,r=matrix,metrics=floats(primary),metrics_rounded=metrics(matrix),baseline_r=baseline_matrix,
        baseline_metrics=floats(base),delta=floats({k:primary[k]-base[k] for k in primary}),
        exact_rational_metrics={k:str(v) for k,v in primary.items()},exact_baseline_metrics={k:str(v) for k,v in base.items()},
        primary_trials=200,epochs=400,teacher_provenance=teachers,mechanism=stages,training_cost=training,
        teacher_build_cost=[json.loads(s) for s in (report/'teacher_build_log.jsonl').read_text().splitlines()],
        run_additional_seeds=additional,postprocess_source_sha256=sha(__file__))
    write(report/'metrics_primary.json',dict(r=matrix,**floats(primary)))
    write(OUT/('RESULT_seed%d.json'%seed),result)
    write(OUT/'RESULT_ARTIFACT_MANIFEST.json',dict(events=events_manifest,trials_sha256=sha(report/'trials_primary.jsonl'),protocol_sha256=sha(OUT/'PROTOCOL.json')))
    write(OUT/('FINAL_VERIFICATION_seed%d.json'%seed),dict(status='PASS',checks=checks))
    if seed==42:write(OUT/'SEED42_DECISION.json',result)
    return result,checks


def main():
    import fcntl
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    incident=OUT/'incidents'/'postprocess_duplicate_r';incident.mkdir(parents=True,exist_ok=True)
    for name in ['QUEUE_STATUS.json','QUEUE.log']:
        p=OUT/name;target=incident/name
        if p.exists() and not target.exists():target.write_bytes(p.read_bytes())
    result,checks=summarize(42)
    # This repair is postprocessing-only: never dispatch training or trials.
    if result['run_additional_seeds']:
        write(OUT/'QUEUE_STATUS.json',dict(status='SEED42_COMPLETE_ADDITIONAL_SEEDS_AUTHORIZED',updated_unix=time.time()))
        return
    checks['conditional_seeds_not_started']=all(not (ROOT/('runs/p1_object_v1/seed%d'%s)).exists() for s in [43,44])
    assert all(checks.values())
    summary=dict(status='COMPLETE',results=[result],seeds_completed=1,primary_gate='NO_OVERALL_BENEFIT',
        seed43_44_not_run=True,e1_enabled=False,baseline_training_rerun=False,secondary_evaluation=False)
    write(OUT/'SUMMARY.json',summary)
    write(OUT/'FINAL_VERIFICATION.json',dict(status='PASS',checks=checks,completed_unix=time.time()))
    write(OUT/'seed42/STATUS.json',dict(status='COMPLETE',seed=42,p1_training_active=False,epochs=400,primary_trials=200))
    write(OUT/'QUEUE_STATUS.json',dict(status='COMPLETE',seeds_completed=1,primary_gate='NO_OVERALL_BENEFIT',updated_unix=time.time(),postprocessing_repaired=True))
    lines=['# P1-OBJECT-v1 最终报告','',
        '**结果：seed42 未通过预注册总体 AUC 筛选；未启动 seed43、44。**','',
        '四阶段各100 epochs全部完成，原主评估200次执行全部完成。E1关闭；β=0.05；每旧窗口16个教师动作；无额外680次辅助评估。','',
        '|指标|P1 seed42|历史 MLR＋IFA seed42|差值（百分点）|','|---|---:|---:|---:|']
    for k in ['auc','fwt','nbt']:lines.append('|%s|%.6f|%.6f|%+.6f|'%(k.upper(),result['metrics'][k],result['baseline_metrics'][k],result['delta'][k]))
    lines+=['','门禁使用原始成功率矩阵按相同公式计算的未舍入值。NBT越低越好，但不能替代预注册的总体AUC门禁。','',
        '|学完阶段|task6|task7|task8|task9|','|---|---:|---:|---:|---:|']
    for i,row in enumerate(result['r']):lines.append('|task%d|%s|'%(i+6,'|'.join('—' if x is None else '%.0f%%'%x for x in row)))
    t=result['training_cost'];tot=t['totals']
    lines+=['','## 实现与实测成本','',
        '教师均为本次P1各自任务ep99输出。781、740、780、660个新目标分阶段入库，最终2961个活动目标；逐阶段核查旧目标记录未刷新，缓存与教师一一对应。',
        '原BC、IFA、优化器和调度次序通过工程验收；原训练、评估源码哈希在完成后再次核验一致。独立评估记录重建10个单元，与原评估日志完全一致。',
        '训练 %.2f 分钟，105500次更新；蒸馏出现在%d步，共%d次旧窗口访问。负MC KL窗口估计%d次（%.5f%%），未截零。'%(t['invocation_seconds']/60,tot['distill_steps'],tot['old_windows'],tot['negative_kl_windows'],100*tot['negative_kl_windows']/tot['old_windows']),
        '蒸馏前向、CPU采样及概率计算累计 %.2f 秒；这不包含其在总反向中的增量开销。峰值训练GPU分配 %.3f GiB。完整进程耗时见EXECUTION_COST.jsonl。'%(tot['distill_forward_seconds'],t['peak_gpu_bytes']/2**30),
        '最终教师数值载荷888300字节，实际教师文件1170892字节，活动索引1940499字节；原活动潜缓存197344635字节。原窗口数未增加，但总存储字节增加。',
        '没有重跑基线用于测速，因此不报告相对基线的训练倍率，也不沿用E1的开销比例。','',
        '## 工程事件与解释边界','',
        '所有测量已正常结束后，原汇总器因dict中重复传入r而停止。原始失败日志已保留。本独立汇总修复只读取现有产物，不重新训练或执行任何rollout；同时明确分离显示舍入与未舍入门禁值。',
        '本结果只支持“当前固定P1配置未通过seed42投入筛选”，不能推广为所有策略分布回放无效。历史同seed基线不是逐位一致的反事实；task6尚无旧教师，不能把该阶段与历史基线的差值归因为P1蒸馏。',
        '本轮不调β、不改变MC数量、不叠加新模块、不改教师质量门、不追加未获门禁授权的seed。']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',metrics=result['metrics'],delta=result['delta'],checks=len(checks)),indent=2))


if __name__=='__main__':main()
