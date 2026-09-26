"""Finalize only after numerical analysis and independent raw-data audit pass."""
import datetime
import os
import re
import shutil
from .common import OUT, read, write
from .analyze import load_csv


def main():
    assert read(OUT/'ANALYSIS_STATUS.json')['status']=='COMPLETE'
    assert read(OUT/'INDEPENDENT_AUDIT.json')['status']=='PASS'
    raw=read(OUT/'RAW_DATA_INDEPENDENT_AUDIT.json');assert raw['status']=='PASS'
    summary=read(OUT/'SUMMARY.json');metrics=read(OUT/'PREDICTION_METRICS.json')
    comp=read(OUT/'PREDICTION_COMPARISONS.json')['comparisons']
    summary['RAW_DATA_AUDIT']='PASS'
    summary['TARGET_VARIATION_STATUS']='NONCONSTANT_OUTCOMES_ON_FIXED_PANEL'
    summary['NEXT_STEP']='No formal training or controller integration; predictive increments not established under the frozen protocol'
    summary['TASK_DELETION_SENSITIVITY']='UNSTABLE_NEAR_CONSTANT_IFA_EXTRAPOLATION'
    resource=read(OUT/'RESOURCE_COST.json')
    matches=re.findall(r'BOOTSTRAP 5000 ([0-9.]+)',(OUT/'ANALYSIS.log').read_text())
    if matches:resource['bootstrap_full_run_seconds']=float(matches[0])
    resource['independent_raw_audit_seconds']=raw['seconds']
    write(OUT/'RESOURCE_COST.json',resource)
    write(OUT/'SUMMARY.json',summary)
    targets=load_csv(OUT/'TARGETS.csv')
    source_levels={(int(r['seed']),int(r['source'])):float(r['P_source_source']) for r in targets}
    source_report=[dict(seed=s,source=t,source_prior_success=v,successes=round(v*20),trials=20) for (s,t),v in sorted(source_levels.items())]
    write(OUT/'SOURCE_MASTERY.json',dict(rows=source_report,zero_success_count=sum(v==0 for v in source_levels.values()),full_success_count=sum(v==1 for v in source_levels.values())))
    extra=['','## 研究结论的直接解释','',
           '1. 固定目标预算的迁移有充分观测变动（-0.80至+0.90），不是所有结局落在同一地板。源任务退化为-0.85至+0.95；不能合并为同一个迁移标签。',
           '2. 迁移预测的最简单难度/掌握度基线M0，OOF MSE=0.183642；IFA=0.195105；动作相容性A=0.188620；latent+A=0.217181；GN+A=0.219004。关系信号没有展示可靠的基线外增量。',
           '3. A相对IFA的MSE差为-0.006485，97.5%区间[-0.069344,0.048603]；仅2/4任务折同向。结论INCONCLUSIVE，不声称A优于IFA。',
           '4. latent+A相对A的MSE差为+0.028561，四个留任务折均变差，但区间跨0。按协议仍记录INCONCLUSIVE；这不构成结构增量支持。GN+A相对latent+A也没有建立迁移预测增益。',
           '5. 源任务损伤预测有描述性正面趋势：A的MSE=0.150028、Spearman=0.6322，M0为0.223842、0.2929。完整区间和seed稳定性尚不足以判SUPPORTED。不要把这个趋势混入迁移结论。',
           '6. M_probe与M0等效，因为首步LR=0，R_old/R_new均为0。这里只能判该冻结探针无信息，不能推断非零适应反馈无效。',
           '7. 删除一个任务后的敏感性分析中，IFA训练相似度近常数而非精确常数，导致极端外推（最大预测绝对值约403447）。未裁剪预测、未调整零方差门，见TASK_DELETION_SENSITIVITY.csv。这是预测设计的局限，不是实际成功率。',
           '8. 不允许据此恢复正式训练、扩建门控或用新的rank/Fisher变体挽救本轮。若继续研究，需要独立协议；本轮主结论是前瞻关系预测增量未建立。',
           '', '## 独立原始数据核验', '',
           '直接重算184320条离线窗口记录（包含共享端点在各对中的重复评价），逐轨迹汇总误差为0；不把这些窗口计作独立关系样本。',
           '逐条检查36分支×300步的batch、LR、优化器和scheduler计数与共享目标参考一致。直接从逐trial数组重算四种标签，最大误差0。',
           '主OOF独立增广最小二乘误差2.22e-15。任务删除的极端预测导致绝对差1.16e-10，但相对差仅1.45e-14；原判据/数据未改变。',
           '', '源模型自身成功率见SOURCE_MASTERY.json；0/20的源任务原样保留，不把无下降空间解释为保护。',
           '', '已完成全部测量、5000次配对重采样和独立核验。formal_training_resumed=false。']
    path=OUT/'FINAL_REPORT.md';text=path.read_text();marker='\n## 研究结论的直接解释'
    text=text.split(marker)[0]+'\n'.join(extra)+'\n'
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(text);os.replace(str(tmp),str(path))
    now=datetime.datetime.now().isoformat()
    write(OUT/'QUEUE_STATUS.json',dict(status='COMPLETE',stage='FINAL_REPORT_COMPLETE',finished_at=now,source_models=12,pairs=36,rollouts=2880,bootstrap_count=5000,formal_training_resumed=False))
    write(OUT/'STATUS.json',dict(protocol_status='COMPLETE',stage='FINAL_REPORT_COMPLETE',formal_training_resumed=False))
    write(OUT/'PROGRESS_CHECK_LATEST.json',dict(checked_at=now,status='COMPLETE',behavior_cells=144,behavior_trials=2880,bootstrap_count=5000,raw_data_audit='PASS',disk_free_gib={d:shutil.disk_usage(d).free/2**30 for d in ['/data','/root']},formal_training_resumed=False))
    print('FINALIZED',now,flush=True)


if __name__=='__main__':main()
