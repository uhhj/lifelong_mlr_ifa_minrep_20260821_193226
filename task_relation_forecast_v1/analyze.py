"""Post-seal analysis only. No models, GPU, training, or signal selection."""
import csv
import io
import os
import time
import datetime
import subprocess
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .common import OUT, TASKS, SEEDS, read, write, sha, freeze
from .statistics import FAMILIES, folds, oof, reconstruct_targets, bootstrap


def table(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO(newline='')
    w = csv.DictWriter(buf, fieldnames=keys); w.writeheader(); w.writerows(rows)
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(buf.getvalue()); os.replace(str(tmp), str(path))


def load_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def metric(y, p):
    y, p = np.asarray(y), np.asarray(p)
    varying = np.ptp(y) > 0 and np.ptp(p) > 0
    return dict(n=len(y), mse=float(np.mean((p-y)**2)), mae=float(np.mean(abs(p-y))),
                spearman=float(spearmanr(y, p).statistic) if varying else None,
                calibration_slope=float(np.dot(p-p.mean(), y-y.mean()) / np.dot(p-p.mean(), p-p.mean())) if np.ptp(p)>0 else None)


def independent_ridge(rows, family, y, train, test):
    # Independently implement transformations and augmented least squares;
    # does not call production design/predict_fold.
    z = np.array([[r[k] for k in FAMILIES[family]] for r in rows], dtype=float)
    cols = []
    for col in z.T:
        vals = col[train]
        cols.append((col-vals.mean())/vals.std() if np.ptp(vals)>0 else np.zeros(len(col)))
    z = np.array(cols).T
    if family in ('M_LA', 'M_GA'):
        col = z[:, -2]*z[:, -1]; vals = col[train]
        z = np.column_stack([z, (col-vals.mean())/vals.std() if np.ptp(vals)>0 else np.zeros(len(col))])
    y = np.asarray(y); yt = y[train]
    if np.ptp(yt)==0:
        return np.full(len(test), yt.mean())
    center = z[train].mean(0); n = len(train); d = z.shape[1]
    lhs = np.vstack([(z[train]-center)/np.sqrt(n), np.eye(d)])
    rhs = np.r_[(yt-yt.mean())/yt.std()/np.sqrt(n), np.zeros(d)]
    beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return ((z[test]-center)@beta)*yt.std()+yt.mean()


def run_bootstrap(rows, success, protocol):
    count, seed = protocol['bootstrap_count'], protocol['bootstrap_seed']
    dest = OUT/'BOOTSTRAP_RESULTS.json'; result = read(dest) if dest.exists() else []
    rng = np.random.RandomState(seed); started = time.monotonic()
    for repeat in range(count):
        rr, yy = [], []
        sampled = rng.choice(SEEDS, len(SEEDS), replace=True)
        for block, original in enumerate(sampled):
            subset = [r for r in rows if r['seed']==original]
            indices = {(int(original), t): rng.randint(0,20,20) for t in TASKS}
            rr.extend(dict(r, bootstrap_block=block) for r in subset)
            yy.extend(reconstruct_targets(subset, success, indices))
        if repeat < len(result):
            continue  # Consume identical RNG sequence when resuming.
        one = dict(repeat=repeat, constant_target_folds=0)
        for target in ('TRANSFER_300', 'DAMAGE_300'):
            y = np.array([r[target] for r in yy])
            for family in FAMILIES:
                pred, meta = oof(rr, family, y)
                one[target+'/'+family] = float(np.mean((pred-y)**2))
                one[target+'/'+family+'/mae'] = float(np.mean(abs(pred-y)))
                one['constant_target_folds'] += sum(m['transform']['constant_target'] for m in meta)
        result.append(one)
        if len(result)%100==0:
            write(dest,result)
            write(OUT/'ANALYSIS_STATUS.json',dict(status='RUNNING',stage='BOOTSTRAP',completed=len(result),total=count,seconds=time.monotonic()-started,formal_training_resumed=False))
            print('BOOTSTRAP',len(result),time.monotonic()-started,flush=True)
    write(dest,result)
    check = bootstrap(rows,success,count=3,seed=seed)
    assert all(abs(result[i][k]-v)<1e-14 for i,r in enumerate(check) for k,v in r.items())
    return result


def main():
    import fcntl
    lock=(OUT/'analysis.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    begin=time.monotonic();protocol=read(OUT/'PROTOCOL.json')
    assert read(OUT/'BEHAVIOR_COMPLETE.json')['trials']==2880
    seal=read(OUT/'SIGNALS_TABLE_SEALED.json');assert sha(OUT/'SIGNALS.csv')==seal['sha256']
    rows=[{k:(int(v) if k in ('seed','source','target') else float(v)) for k,v in r.items()} for r in load_csv(OUT/'SIGNALS.csv')]
    assert len(rows)==36 and len({(r['seed'],r['source'],r['target']) for r in rows})==36
    freeze(OUT/('analysis_implementations/'+sha(__file__)+'.json'),dict(protocol_sha256=sha(OUT/'PROTOCOL.json'),signal_sha256=seal['sha256'],code_sha256=sha(__file__),bootstrap_count=5000,bootstrap_seed=protocol['bootstrap_seed'],primary_interval=.975,secondary_interval=.95,acceptance='negative MSE difference, interval entirely negative, >=3/4 task folds and all3 seed folds negative; otherwise no support or inconclusive',seed_consistency_interpretation='all3 is a conservative operationalization of not driven by one seed; full fold values reported',formal_training_resumed=False))
    success={};rollouts=[];behavior_seconds=0.;audit={};complete=[]
    for q in sorted((OUT/'behavior').glob('s*/*/task*/COMPLETE.json')):
        c=read(q);assert c['status']=='PASS' and c['formal_training_resumed'] is False
        for name,h in c['files'].items():assert sha(q.parent/name)==h,(q,name)
        ts=read(q.parent/'TRIALS.json');assert len(ts)==20 and sorted(t['trial'] for t in ts)==list(range(20))
        key=(c['cell']['seed'],c['cell']['model_id'],c['cell']['task']);assert key not in success
        success[key]=np.array([t['success'] for t in sorted(ts,key=lambda t:t['trial'])]);assert set(success[key])<={0,1}
        assert int(success[key].sum())==c['successes']
        rollouts.extend(dict(t,cell_seconds=c['seconds'],checkpoint_sha256=c['checkpoint_sha256']) for t in ts)
        behavior_seconds+=c['seconds'];complete.append(q)
    assert len(success)==144 and len(rollouts)==2880
    table(OUT/'ROLLOUTS.csv',rollouts)
    targets=reconstruct_targets(rows,success);targetrows=[]
    for row, labels in zip(rows,targets):
        seed,s,t=row['seed'],row['source'],row['target'];pr=lambda m,e:float(success[seed,m,e].mean())
        vals=dict(P_base_target=pr('base',t),P_source_source=pr('source%d'%s,s),P_source_target=pr('source%d'%s,t),P_reference_target=pr('reference%d'%t,t),P_transfer_source=pr('transfer%d_%d'%(s,t),s),P_transfer_target=pr('transfer%d_%d'%(s,t),t))
        assert labels['TRANSFER_300']==vals['P_transfer_target']-vals['P_reference_target']
        assert labels['DAMAGE_300']==vals['P_source_source']-vals['P_transfer_source']
        targetrows.append(dict(seed=seed,source=s,target=t,**labels,**vals,target_own_improvement=vals['P_transfer_target']-vals['P_source_target'],reference_own_improvement=vals['P_reference_target']-vals['P_base_target']))
    table(OUT/'TARGETS.csv',targetrows)
    training=[];offline=[];parities={};seconds={};raw_nll_errors=[]
    for stage in ['sources','transfers']:
        paths=sorted((OUT/stage).glob('*/COMPLETE.json'));assert len(paths)==(12 if stage=='sources' else 36)
        seconds[stage]=sum(read(q)['seconds'] for q in paths)
        for q in paths:
            raw=[read(f) for f in sorted((q.parent/'steps').glob('*.json'))]
            assert len(raw)==read(q)['steps']
            assert [v['step'] for v in raw]==list(range(1,len(raw)+1))
            training.extend(dict(r,stage=stage,unit=q.parent.name) for r in raw)
    assert len(training)==22800
    for row in rows:
        unit='s%d_%d_%d'%(row['seed'],row['source'],row['target'])
        sp=OUT/'signals'/unit;tp=OUT/'transfers'/unit;op=OUT/'offline'/unit
        ss=read(sp/'SEALED.json');assert sha(sp/'SIGNALS.json')==ss['signals_sha256'];assert read(sp/'SIGNALS.json')['scores']==row
        assert read(tp/'COMPLETE.json')['signal_seal_sha256']==sha(sp/'SEALED.json')
        assert read(op/'COMPLETE.json')['signal_seal_sha256']==sha(sp/'SEALED.json')
        parity=read(tp/'STREAM_PARITY.json');assert parity['status']=='PASS';parities[unit]=parity
        offline.extend(load_csv(op/'OFFLINE_NLL.csv'))
    table(OUT/'TRAINING_PROGRESS.csv',training);table(OUT/'OFFLINE_NLL.csv',offline);write(OUT/'STREAM_PARITY.json',parities)
    seconds['signals']=sum(read(q)['seconds'] for q in (OUT/'signals').glob('*/SIGNALS.json'))
    seconds['valid_behavior']=behavior_seconds
    target_names=['TRANSFER_300','DAMAGE_300','TRANSFER_0','ACQUISITION_DIFF_300']
    predictions={};metrics={};oofrows=[];transforms={};max_independent_error=0.
    for mode in ['task','seed']:
        for target in target_names:
            y=np.array([r[target] for r in targets])
            for family in FAMILIES:
                key=mode+'/'+target+'/'+family;p,meta=oof(rows,family,y,mode);predictions[key]=p;transforms[key]=meta
                groups={'all':list(range(36))}
                groups.update({'target%d'%t:[i for i,r in enumerate(rows) if r['target']==t] for t in TASKS})
                groups.update({'seed%d'%s:[i for i,r in enumerate(rows) if r['seed']==s] for s in SEEDS})
                metrics[key]={k:metric(y[ix],p[ix]) for k,ix in groups.items()}
                for f in folds(rows,mode):
                    pi=independent_ridge(rows,family,y,f['train'],f['test'])
                    err=float(np.max(abs(pi-p[f['test']])));max_independent_error=max(max_independent_error,err)
                for i,r in enumerate(rows):oofrows.append(dict(seed=r['seed'],source=r['source'],target=r['target'],mode=mode,outcome=target,family=family,observed=float(y[i]),predicted=float(p[i])))
    assert max_independent_error<1e-12,max_independent_error
    table(OUT/'OOF_PREDICTIONS.csv',oofrows);write(OUT/'PREDICTION_METRICS.json',metrics);write(OUT/'FOLD_TRANSFORMS.json',transforms)
    boot=run_bootstrap(rows,success,protocol)
    comparisons=[]
    pairs=list(dict.fromkeys([(m,b) for m in FAMILIES for b in ['M0','M_A','M_IFA'] if m!=b]+[('M_GA','M_LA')]))
    for outcome in ['TRANSFER_300','DAMAGE_300']:
        for candidate,baseline in pairs:
            primary=outcome=='TRANSFER_300' and (candidate,baseline) in [('M_A','M_IFA'),('M_LA','M_A')]
            level=.975 if primary else .95;q=(1-level)/2
            a=metrics['task/'+outcome+'/'+candidate];b=metrics['task/'+outcome+'/'+baseline]
            deltas=[z[outcome+'/'+candidate]-z[outcome+'/'+baseline] for z in boot]
            ci=np.quantile(deltas,[q,1-q]).tolist()
            fold={str(t):a['target%d'%t]['mse']-b['target%d'%t]['mse'] for t in TASKS}
            seeds={str(s):metrics['seed/'+outcome+'/'+candidate]['seed%d'%s]['mse']-metrics['seed/'+outcome+'/'+baseline]['seed%d'%s]['mse'] for s in SEEDS}
            diff=a['all']['mse']-b['all']['mse'];supported=diff<0 and ci[1]<0 and sum(v<0 for v in fold.values())>=3 and all(v<0 for v in seeds.values())
            verdict='SUPPORTED' if supported else ('NOT_SUPPORTED' if ci[0]>0 else 'INCONCLUSIVE')
            comparisons.append(dict(outcome=outcome,candidate=candidate,baseline=baseline,primary=primary,mse_difference=diff,mae_difference=a['all']['mae']-b['all']['mae'],interval_level=level,mse_interval=ci,mae_interval=np.quantile([z[outcome+'/'+candidate+'/mae']-z[outcome+'/'+baseline+'/mae'] for z in boot],[q,1-q]).tolist(),task_fold_mse_difference=fold,leave_seed_mse_difference=seeds,verdict=verdict))
    write(OUT/'PREDICTION_COMPARISONS.json',dict(comparisons=comparisons,bootstrap_count=len(boot),constant_target_fold_evaluations=sum(r['constant_target_folds'] for r in boot),scope='fixed four-task panel; three optimization seeds; paired evaluation initial states'))
    selections=[]
    for family in FAMILIES:
        p=predictions['task/TRANSFER_300/'+family]
        for seed in SEEDS:
            for t in TASKS:
                ix=[i for i,r in enumerate(rows) if r['seed']==seed and r['target']==t];best=max(p[i] for i in ix);chosen=[i for i in ix if p[i]==best]
                vals=[targetrows[i]['P_transfer_target'] for i in ix]
                selected=float(np.mean([targetrows[i]['P_transfer_target'] for i in chosen]))
                selections.append(dict(seed=seed,target=t,family=family,tied_selected_sources=[rows[i]['source'] for i in chosen],selected_success=selected,uniform_source_success=float(np.mean(vals)),selected_minus_uniform=selected-float(np.mean(vals)),oracle_success=max(vals),regret=max(vals)-selected))
    write(OUT/'SOURCE_SELECTION.json',dict(rows=selections,ties='average all exactly tied sources; no outcome-dependent tie break',status='DESCRIPTIVE'))
    variation={k:dict(min=float(min(r[k] for r in targets)),max=float(max(r[k] for r in targets)),std=float(np.std([r[k] for r in targets])),unique=len(set(r[k] for r in targets))) for k in target_names}
    def verdict(c,b,o='TRANSFER_300'):return next(x['verdict'] for x in comparisons if (x['candidate'],x['baseline'],x['outcome'])==(c,b,o))
    audit.update(status='PASS',behavior_cells=144,rollouts=2880,training_updates=len(training),signals=36,signal_table_hash=sha(OUT/'SIGNALS.csv'),all_completed_behavior_file_hashes_match=True,independent_augmented_lstsq_max_error=max_independent_error,bootstrap_first3_match_frozen_implementation=True,formal_training_resumed=False)
    write(OUT/'INDEPENDENT_AUDIT.json',audit)
    seconds['analysis']=time.monotonic()-begin
    write(OUT/'RESOURCE_COST.json',dict(measured_seconds=seconds,source_updates=12000,target_updates=10800,shadow_updates=36,valid_rollouts=2880,excluded='cache/bootstrap startup, discarded pairing diagnostics, failed attempts and human/debug pauses are not included in phase sums',formal_training_resumed=False))
    summary=dict(protocol_status='COMPLETE',MEASUREMENT_VALIDITY='PASS_WITH_DISCLOSED_PROVENANCE_LIMITATIONS',TARGET_VARIATION=variation,ACTION_AFFINITY_PREDICTIVE_VALUE=verdict('M_A','M_IFA'),STRUCTURE_PLUS_ACTION_INCREMENT=verdict('M_LA','M_A'),GN_INCREMENT_OVER_CHEAP_STRUCTURE=verdict('M_GA','M_LA'),ONE_STEP_PROBE_REFERENCE_VALUE='UNINFORMATIVE_ZERO_FIRST_STEP_LR',TRANSFER_FORECAST={m:verdict(m,'M0') for m in FAMILIES if m!='M0'},OLD_DAMAGE_FORECAST={m:verdict(m,'M0','DAMAGE_300') for m in FAMILIES if m!='M0'},GENERALIZATION_SCOPE='Four fixed tasks, 12 ordered pairs, 3 optimization seeds sharing one pretrained base; no population-level or native-MLR controller conclusion',RESOURCE_COST=seconds,diagnostic_training_performed=True,formal_training_resumed=False)
    write(OUT/'SUMMARY.json',summary)
    lines=['# 任务关系信号前瞻预测实验 v1：最终报告','', '完成12个源模型、36个有序seed×任务对、22800次诊断更新和144×20=2880次有效闭环。正式lifelong训练未恢复。','', '## 冻结的主比较','', '|目标|候选−基线|OOF MSE差|区间|4任务折同向数|判定|','|---|---|---:|---|---:|---|']
    for c in comparisons:
        if c['primary'] or (c['candidate'],c['baseline'])==('M_GA','M_LA'):
            lines.append('|%s|%s−%s|%.6g|%s %.6g, %.6g|%d/4|%s|'%(c['outcome'],c['candidate'],c['baseline'],c['mse_difference'],c['interval_level'],*c['mse_interval'],sum(v<0 for v in c['task_fold_mse_difference'].values()),c['verdict']))
    lines+=['','## 全模型的严格留任务结果','','|目标|模型|MSE|MAE|Spearman|','|---|---|---:|---:|---:|']
    for outcome in target_names:
        for family in FAMILIES:
            m=metrics['task/'+outcome+'/'+family]['all'];lines.append('|%s|%s|%.6g|%.6g|%s|'%(outcome,family,m['mse'],m['mae'],m['spearman']))
    lines+=['','## 结论与边界','', '```json', __import__('json').dumps(summary,ensure_ascii=False,indent=2), '```','', 'M_probe的R_old/R_new全部为零：保留原生第一步LR=0，没有换后续步制造信号，因此本轮不能判断非零影子适应是否有价值。','', '主区间由5000次seed整块＋共享初态配对重采样、重新拟合产生；两个TRANSFER主比较各用97.5%区间。其他比较95%且属次级描述，不作多重比较后胜者搜索。', '', '原官方基座配套训练配方与检查点来源一致，但原作者训练job日志不可得，暴露核验是来源证据而非直接原日志验证。', '', '行为配对曾因非策略输入的辅助四元数字段微小变化误拦；保留失败证据，仅修正工程检查范围。有效结果要求初始状态、原生策略输入和RNG配对检查通过。无阈值或试验预算调节。', '', '有效rollout每模型任务仅20次；单个trial差异不自动构成正负迁移证据。36行只有12种有序关系，三个seed不是独立预训练种子。', '', '本轮无回放、1000步源预算/300步目标预算，不代表完整MLR或终身训练收益。GN未显示相对廉价结构增量时，不扩展rank/Fisher挽救；无可靠关系增量时不扩建控制器。', '', '逐次结果见ROLLOUTS.csv，绝对成功率/条件迁移/源损伤见TARGETS.csv；离线逐轨迹均值见OFFLINE_NLL.csv；逐窗口原始值保留在offline/*/WINDOW_NLL.csv。', '', '统计正确性：独立增广最小二乘重算全部OOF预测，最大误差 %.3g；原始成功数组独立重算标签；固定bootstrap前三次与冻结实现一致。'%max_independent_error,'', 'formal_training_resumed=false']
    text='\n'.join(lines)+'\n';tmp=OUT/'FINAL_REPORT.md.tmp';tmp.write_text(text);os.replace(str(tmp),str(OUT/'FINAL_REPORT.md'))
    write(OUT/'ANALYSIS_STATUS.json',dict(status='COMPLETE',seconds=time.monotonic()-begin,formal_training_resumed=False))
    print('ANALYSIS_COMPLETE',summary,flush=True)


if __name__=='__main__':
    main()
