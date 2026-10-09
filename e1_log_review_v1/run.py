"""Summarize actual accepted displacement, residual excess, and buffer coverage.

Uses only saved JSON/JSONL and source text. All thresholds below are reporting
bins, not new acceptance rules. No models, tensors, training, or evaluation.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(t,p)


def table(p, rows):
    if not rows:return
    t=p.with_name(p.name+'.tmp')
    with t.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(t,p)


def quantile(xs, q):
    ys=sorted(xs); pos=(len(ys)-1)*q; a=int(pos); b=min(a+1,len(ys)-1)
    return ys[a]+(ys[b]-ys[a])*(pos-a)


def stats(xs):
    xs=list(xs)
    if not xs:return {'n':0}
    assert all(math.isfinite(x) for x in xs)
    return dict(n=len(xs),mean=statistics.mean(xs),min=min(xs),
                p10=quantile(xs,.1),median=quantile(xs,.5),p90=quantile(xs,.9),
                p95=quantile(xs,.95),p99=quantile(xs,.99),max=max(xs))


def amplitude_and_residual(rows):
    a=[r for r in rows if r['accepted']]
    pos=[r for r in a if r['residual']>0]
    values=['delta_norm','correction_norm','actual_correction_norm','actual_to_native',
            'intended_to_native','actual_to_intended','q','excess','epsilon','xi',
            'loss_decrease','final_minus_initial','residual','residual_to_excess',
            'residual_to_epsilon','removed_excess_fraction','improvement_to_excess']
    return dict(checks=len(rows),triggered=sum(r['triggered'] for r in rows),accepted=len(a),
        rejected=sum(r['triggered'] and not r['accepted'] for r in rows),
        capped=sum(r['capped'] for r in rows),accepted_stats={k:stats(r[k] for r in a) for k in values},
        accepted_ratio_bins={label:sum(lo<=r['actual_to_native']<hi for r in a) for label,lo,hi in
            [('below_0.01pct',0,.0001),('0.01_to_0.1pct',.0001,.001),('0.1_to_1pct',.001,.01),('1_to_5pct',.01,.05),('5_to_25pct',.05,.25),('at_least_25pct',.25,math.inf)]},
        residual_positive_count=len(pos),residual_positive_fraction=len(pos)/len(a) if a else None,
        residual_above_xi_count=sum(r['residual']>r['xi'] for r in a),
        residual_above_epsilon_count=sum(r['residual']>r['epsilon'] for r in a),
        final_above_initial_count=sum(r['final_minus_initial']>0 for r in a),
        accepted_zero_actual_correction_count=sum(r['actual_correction_norm']==0 for r in a),
        positive_residual_stats=stats(r['residual'] for r in pos),
        residual_ratio_positive_only=stats(r['residual_to_excess'] for r in pos),
        total_excess=sum(r['excess'] for r in a),total_residual=sum(r['residual'] for r in a),
        excess_weighted_removed_fraction=1-sum(r['residual'] for r in a)/sum(r['excess'] for r in a) if a else None)


def analyze(root, output):
    source=root/'reports/e1_object_v1'; seed_dir=source/'seed42'
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'RESULT_ARTIFACT_MANIFEST.json').read_text())
    hashes={r['path']:r['sha256'] for r in manifest['files']}
    inputs={};all_rows=[];stage_results={};sample_rows=[];epoch_rows=[];task_rows=[]
    checks=dict(source_event_hashes=True,unique_event_identity=True,check_schedule=True,
        legal_samples=True,no_within_batch_duplicate=True,independent_rng_reproduced=True,
        loss_formulas_match=True,acceptance_rule_matches=True,aggregate_counts_match=True)
    for task in [6,7,8,9]:
        bp=seed_dir/('BUFFER_BEFORE_TASK%d.json'%task);inputs[str(bp.relative_to(root))]=sha(bp)
        universe=[Path(x).name for x in json.loads(bp.read_text())['files']]
        assert len(set(universe))==len(universe)
        files=sorted((seed_dir/'events').glob('task%d_epoch*.jsonl'%task));rows=[]
        for f in files:
            if '.attempt' in f.name:continue
            digest=sha(f);rel=str(f.relative_to(root));inputs[rel]=digest
            assert hashes[rel]==digest
            for line in f.read_text().splitlines():
                r=json.loads(line);assert r['task']==task and r['seed']==42
                r['check_index']=len(rows)+1; rows.append(r)
        assert len({r['update'] for r in rows})==len(rows)
        assert [r['update'] for r in rows]==list(range(10,10*(len(rows)+1),10))
        rng=random.Random(1000003+42); counts=Counter();accepted_counts=Counter();visits={x:[] for x in universe}
        milestones={};seen=set();intersections=[];previous=set();old_task_counts=Counter(int(x.split('__')[0][1:]) for x in universe)
        for r in rows:
            ids=r['sample_ids'];assert len(ids)==min(20,len(universe)) and len(set(ids))==len(ids)
            assert set(ids)<=set(universe) and all(int(x.split('__')[0][1:])<task for x in ids)
            expected=[universe[i] for i in rng.sample(range(len(universe)),min(20,len(universe)))]
            assert ids==expected
            counts.update(ids)
            if r['accepted']:accepted_counts.update(ids)
            for x in ids:visits[x].append(r['check_index'])
            seen.update(ids)
            for percent in [50,90,95,99,100]:
                if str(percent) not in milestones and len(seen)>=math.ceil(len(universe)*percent/100):
                    milestones[str(percent)]=dict(check_index=r['check_index'],optimizer_update=r['update'],epoch=r['epoch'])
            if previous:intersections.append(len(previous & set(ids)))
            previous=set(ids)
            assert math.isclose(r['q'],r['Lc']-r['L0'],rel_tol=1e-12,abs_tol=1e-12)
            assert math.isclose(r['excess'],max(0,r['q']-r['epsilon']),rel_tol=1e-12,abs_tol=1e-12)
            assert r['triggered']==(r['q']>r['epsilon'])
            if r['Lf'] is not None:assert r['accepted']==(r['Lf']<=r['Lc']-r['xi'])
            if r['accepted']:
                r.update(actual_to_native=r['actual_correction_norm']/r['delta_norm'],
                    intended_to_native=r['correction_norm']/r['delta_norm'],
                    actual_to_intended=r['actual_correction_norm']/r['correction_norm'],
                    final_minus_initial=r['Lf']-r['L0'],loss_decrease=r['Lc']-r['Lf'],
                    residual=max(0,r['Lf']-r['L0']-r['epsilon']))
                assert math.isclose(r['residual'],r['residual_excess'],rel_tol=1e-12,abs_tol=1e-12)
                r.update(residual_to_excess=r['residual']/r['excess'],residual_to_epsilon=r['residual']/r['epsilon'],
                    removed_excess_fraction=1-r['residual']/r['excess'],improvement_to_excess=r['loss_decrease']/r['excess'])
        cov=dict(buffer_windows=len(universe),draws=sum(counts.values()),unique_checked=len(counts),
            unique_in_accepted_batches=len(accepted_counts),coverage_fraction=len(counts)/len(universe) if universe else None,
            accepted_batch_coverage_fraction=len(accepted_counts)/len(universe) if universe else None,
            per_check_buffer_fraction=min(20,len(universe))/len(universe) if universe else None,
            expected_unique_uniform=len(universe)*(1-(1-min(20,len(universe))/len(universe))**len(rows)) if universe else None,
            sample_frequency=stats(counts[x] for x in universe),accepted_batch_frequency=stats(accepted_counts[x] for x in universe),
            full_coverage_milestones=milestones,adjacent_check_intersection=stats(intersections),
            expected_adjacent_check_intersection=400/len(universe) if universe else None)
        stage_results[str(task)]=dict(amplitude_residual=amplitude_and_residual(rows),coverage=cov)
        for old,n in sorted(old_task_counts.items()):
            members=[x for x in universe if int(x.split('__')[0][1:])==old];member_set=set(members);appearances=sum(counts[x] for x in members)
            tr=dict(stage=task,old_task=old,buffer_windows=n,expected_draw_fraction=n/len(universe),draws=appearances,
                actual_draw_fraction=appearances/sum(counts.values()),unique_checked=sum(counts[x]>0 for x in members),
                unique_in_accepted_batches=sum(accepted_counts[x]>0 for x in members),
                checks_without_this_task=sum(not any(x in member_set for x in r['sample_ids']) for r in rows))
            task_rows.append(tr)
        for x in universe:
            v=visits[x];sample_rows.append(dict(stage=task,sample_id=x,old_task=int(x.split('__')[0][1:]),checks=counts[x],
                accepted_batch_checks=accepted_counts[x],first_check=v[0] if v else None,last_check=v[-1] if v else None,
                max_between_check_gap=max((b-a for a,b in zip(v,v[1:])),default=None)))
        for epoch in range(100):
            er=[r for r in rows if r['epoch']==epoch];a=[r for r in er if r['accepted']];seen_e=set(x for r in er for x in r['sample_ids'])
            epoch_rows.append(dict(stage=task,epoch=epoch,checks=len(er),accepted=len(a),unique_checked=len(seen_e),
                buffer_coverage=len(seen_e)/len(universe) if universe else None,
                median_actual_ratio=statistics.median(r['actual_to_native'] for r in a) if a else None,
                residual_positive=sum(r['residual']>0 for r in a),residual_above_xi=sum(r['residual']>r['xi'] for r in a),
                median_residual=statistics.median(r['residual'] for r in a) if a else None))
        cov['per_epoch_coverage']=stats(r['buffer_coverage'] for r in epoch_rows if r['stage']==task and r['buffer_coverage'] is not None)
        all_rows.extend(rows)
    original=json.loads((source/'SUMMARY.json').read_text())['results'][0]['training_cost']['totals']
    assert len(all_rows)==original['checks'] and sum(r['accepted'] for r in all_rows)==original['accepted']
    assert sum(r['triggered'] for r in all_rows)==original['triggered']
    accepted=[r for r in all_rows if r['accepted']]
    fields=['task','epoch','update','check_index','L0','Lc','Lf','epsilon','xi','q','excess','delta_norm','grad_norm',
        'correction_norm','actual_correction_norm','actual_to_native','intended_to_native','actual_to_intended',
        'loss_decrease','final_minus_initial','residual','residual_to_excess','residual_to_epsilon','removed_excess_fraction','improvement_to_excess']
    table(output/'ACCEPTED_CORRECTIONS.csv',[{k:r[k] for k in fields} for r in accepted])
    table(output/'SAMPLE_COVERAGE.csv',sample_rows);table(output/'EPOCH_SUMMARY.csv',epoch_rows);table(output/'TASK_COVERAGE.csv',task_rows)
    summary=dict(status='COMPLETE',scope='existing seed42 E1 logs only',training_run=False,evaluation_run=False,
        log_integrity_checks=checks,all_stages=amplitude_and_residual(all_rows),stages=stage_results,
        task_coverage=task_rows,
        interpretation_limits=['actual_correction_norm is the logged FP32 norm of parameter difference after materialization, not a recomputed FP64 vector norm',
        'actual_to_intended compares norms only; no correction vectors were logged, so vector error/cosine/cumulative drift are unavailable',
        'residual positive follows the frozen exact comparator; above_xi and above_epsilon are descriptive bins, not relaxed gates',
        'sum of excess/residual is across different batches and model states; it is not cumulative old-task forgetting',
        'sample coverage is over legal buffer window IDs; trajectory identities, independent old windows and per-window post-correction losses were not logged',
        'all-window coverage over a whole stage does not mean simultaneous or persistent protection',
        'coverage in accepted batches does not establish benefit to each window because the acceptance loss is a batch mean'])
    write(output/'SUMMARY.json',summary)
    for rel in ['e1_object_v1/control.py','e1_object_v1/runtime.py','reports/e1_object_v1/PROTOCOL.json','reports/e1_object_v1/SUMMARY.json']:
        inputs[rel]=sha(root/rel)
    write(output/'INPUT_MANIFEST.json',dict(files=inputs,analysis_source_sha256=sha(Path(__file__)),source_root=str(root)))
    def fmt(v):return '—' if v is None else '%.6g'%v
    all_a=summary['all_stages'];s=all_a['accepted_stats']
    lines=['# E1 已接受修正与检查覆盖复盘','', '仅使用已有 seed42 日志；未加载模型、未训练或重新评估。',
        '', '共复核%d次检查、%d次触发、%d次接受。已接受修正实际/原生位移比中位数%.4f%%，P90为%.4f%%，最大%.4f%%；没有触及25%%上限，也没有已接受修正的实际范数为零。'%(all_a['checks'],all_a['triggered'],all_a['accepted'],100*s['actual_to_native']['median'],100*s['actual_to_native']['p90'],100*s['actual_to_native']['max']),
        '', '%d/%d（%.3f%%）接受事件仍有正残余，其中%d次残余大于原xi。修正前超标量加权消除了%.3f%%；逐次消除比例中位数为%.3f%%。接受表示相对候选可靠改善，不能表述成“旧损失未增加”。'%(all_a['residual_positive_count'],all_a['accepted'],100*all_a['residual_positive_fraction'],all_a['residual_above_xi_count'],100*all_a['excess_weighted_removed_fraction'],100*s['removed_excess_fraction']['median']),
        '', '实际/计划修正范数比中位数%.9f，P10/P90为%.9f / %.9f，范围%.9f–%.9f。这些长度记录不支持“普遍被FP32写回吞掉”，但未保存向量，不能据此认证方向误差。'%(s['actual_to_intended']['median'],s['actual_to_intended']['p10'],s['actual_to_intended']['p90'],s['actual_to_intended']['min'],s['actual_to_intended']['max']),
        '', '## 已接受修正的幅度', '', '|阶段|检查/接受|实际幅度比中位数|P90|最大值|触及限幅次数|', '|---|---:|---:|---:|---:|---:|']
    for t in ['7','8','9']:
        a=stage_results[t]['amplitude_residual'];s=a['accepted_stats']['actual_to_native']
        lines.append('|%s|%d / %d|%.4f%%|%.4f%%|%.4f%%|%d|'%(t,a['checks'],a['accepted'],100*s['median'],100*s['p90'],100*s['max'],a['capped']))
    lines+=['','幅度比为日志中实际参数差范数 / 原生 AdamW 位移范数。只统计接受后保留的修正；被撤销的尝试不计入实际干预。',
        '', '## 接受后的残余超标', '', '残余 = max(0, Lf − L0 − epsilon)。通过接受条件仅表示 Lf ≤ Lc − xi，不要求残余为零。',
        '', '|阶段|残余>0 / 接受数|残余>xi|残余>epsilon|残余中位数（含零）|超标量加权消除比例|', '|---|---:|---:|---:|---:|---:|']
    for t in ['7','8','9']:
        a=stage_results[t]['amplitude_residual']
        lines.append('|%s|%d / %d|%d|%d|%.6g|%.3f%%|'%(t,a['residual_positive_count'],a['accepted'],a['residual_above_xi_count'],a['residual_above_epsilon_count'],a['accepted_stats']['residual']['median'],100*a['excess_weighted_removed_fraction']))
    lines+=['','加权比例为 1−sum(残余)/sum(修正前超标量)，只描述本次抽样批次纠偏，不能解释为累计遗忘降低。',
        '', '## 检查样本覆盖', '', '|阶段|合法旧窗口|每次检查占比|总抽取次数|覆盖窗口|出现在接受批次中的窗口|每窗口频次 中位数 / 范围|首次100%覆盖（epoch/check）|', '|---|---:|---:|---:|---:|---:|---:|---|']
    for t in ['7','8','9']:
        c=stage_results[t]['coverage'];s=c['sample_frequency'];m=c['full_coverage_milestones'].get('100',{})
        lines.append('|%s|%d|%.3f%%|%d|%d|%d|%.1f / %d–%d|%s / %s|'%(t,c['buffer_windows'],100*c['per_check_buffer_fraction'],c['draws'],c['unique_checked'],c['unique_in_accepted_batches'],s['median'],s['min'],s['max'],m.get('epoch','—'),m.get('check_index','—')))
    lines+=['','task6 的初始缓存为空，零次 E1 检查。其后只检查本次运行当时的合法旧缓存，不涉及预训练任务0–5或未来任务。全部抽样顺序均从冻结的独立 RNG 重建并逐项吻合；每批20个窗口，无批内重复或越界样本。',
        '', '单epoch缓存覆盖率中位数：task7 %.2f%%、task8 %.2f%%、task9 %.2f%%。首次100%%覆盖表中的epoch为0-based。'%(100*stage_results['7']['coverage']['per_epoch_coverage']['median'],100*stage_results['8']['coverage']['per_epoch_coverage']['median'],100*stage_results['9']['coverage']['per_epoch_coverage']['median']),
        '', '## 解释边界','', '累计覆盖不等于每一步覆盖：E1只每10次原生更新检查一次，且每次只约束抽中批次的平均NLL。日志不含逐样本修正后损失或轨迹ID，因此无法据此认定全部窗口逐一受益、跨轨迹覆盖充分或独立旧任务得到保护。',
        '', '没有出现某些合法缓存窗口从未被检查的情况；也没有限幅频繁阻止修正的情况。按冻结公式，单独提高rho不会改变这批未触及上限的修正。修正幅度小本身既不证明无效，也不证明是闭环失败的原因。',
        '', '实际/意图范数之比只比较长度，不能推断两修正向量的夹角或FP32向量误差。残余很小的符号还可能受已记录FP32损失精度影响，本报告同时给出相对于原xi的分层，不修改原接受规则。',
        '', '这些是日志总体的描述性统计，不把检查批次数当独立实验样本，也不将它们取代已经失败的闭环AUC主判定。',
        '', '完整分位数与按任务比例见 SUMMARY.json；逐修正、逐窗口、逐epoch数据见同目录CSV。']
    (output/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary['all_stages'],'stages':stage_results,'task_coverage':task_rows},ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args()
    analyze(args.root,args.root/'reports/e1_object_v1/log_review_v1')
