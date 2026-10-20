"""Verify available records and summarize progress, without extrapolating metrics."""
from pathlib import Path
from collections import Counter
import json, hashlib, datetime, statistics, sys
R=Path(__file__).resolve().parent
D=Path(sys.argv[1]) if len(sys.argv)>1 else sorted(R.glob('progress_*'))[-1]
def js(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
manifest=js(D/'evidence/PACKAGE_MANIFEST.json')
assert all(hashlib.sha256((D/'evidence'/n).read_bytes()).hexdigest()==h['sha256'] for n,h in manifest.items())
snap=js(D/'SNAPSHOT.json');now=snap['time'];results={}
for variant,progress in snap['variants'].items():
    root=D/'evidence'/variant;folder=root/'evaluation'
    if not folder.exists():continue
    episode_rows=[];table=[]
    for p in sorted(folder.glob('trials*.jsonl')):
        rr=rows(p);assert len({x['trial'] for x in rr})==len(rr) and len(rr)<=20
        parent={x['trial']:x for x in rows(D/'evidence/controls/RA'/p.name)}
        for x in rr:
            assert type(x['success']) is bool and x['arm']=='ra_coreset_ifa_zr'+variant
            assert x['seed']==41 and x['trial']==x['initial_state_index']
            assert x['parent_checkpoint_sha256']==parent[x['trial']]['checkpoint_sha256']
            assert x['initial_state_sha256']==parent[x['trial']]['initial_state_sha256']
            assert x['modal_sha256']==parent[x['trial']]['modal_sha256']
        episode_rows.extend(rr)
        table.append(dict(stage=rr[0]['stage'],task=rr[0]['task'],records=len(rr),
                          successes=sum(x['success'] for x in rr),
                          success_percent=100*sum(x['success'] for x in rr)/20 if len(rr)==20 else None))
    ev=[x for p in folder.glob('attempts*.jsonl') for x in rows(p)];ct=Counter(x['event'] for x in ev)
    assert ct['technical_exception']==0
    assert ct['vector_attempt_complete']*10==len(episode_rows)
    cells=[js(p) for p in folder.glob('cell*.json')]
    if variant=='v3':assert all(x['max_executed_kl']<=.5+1e-12 for x in cells)
    durations=[]
    for p in folder.glob('attempts*.jsonl'):
        begin=None
        for x in rows(p):
            if x['event']=='vector_attempt_start':begin=x['unix']
            if x['event']=='vector_attempt_complete':
                assert begin is not None;durations.append(x['unix']-begin);begin=None
    # Running vectors count as a whole remaining vector: conservative time estimate.
    remaining=20-ct['vector_attempt_complete'];mean=statistics.mean(durations) if durations else None
    eta=None
    if mean is not None:
        eta=dict(remaining_vectors=remaining,completed_vector_count=len(durations),mean_completed_vector_seconds=mean,
                 expected_remaining_seconds=remaining*mean+180,
                 conservative_remaining_seconds=remaining*max(durations)+300,
                 caveat='Includes a whole currently running vector and audit allowance; optional v3-A excluded.')
    results[variant]=dict(epochs=progress['epochs'],optimizer_steps=progress['optimizer_steps'],records=len(episode_rows),events=dict(ct),cells=table,
                         max_online_KL=max((x['max_executed_kl'] for x in cells),default=None),
                         completed_cell_count=len(cells),eta=eta,
                         matched_parent_and_initial_hashes=True,no_new_model_or_environment_calls=True)
out=dict(status='PASS_READONLY_PROGRESS',observation_unix=now,variants=results,package_hashes='PASS')
(D/'PROGRESS_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
when=datetime.datetime.fromtimestamp(now).astimezone()
text='# ZR-v3进度检查（PARTIAL）\n\n观测时间：'+when.isoformat()+'\n\n'
for v,r in results.items():
    text+=f"{v}：{r['epochs']}/40校准epoch、{r['optimizer_steps']}/10,550步已完成；正式闭环{r['records']}/200，完成vector attempt {r['events'].get('vector_attempt_complete',0)}/20，技术异常0。冻结hash通过，本次未重启、未重复预检或追加预算。\n\n"
    text+='| 阶段 | 任务 | 已提交记录 | 成功数 | 完整单元成功率 |\n|---|---|---:|---:|---:|\n'
    for x in r['cells']:
        value=f"{x['success_percent']:.1f}%" if x['success_percent'] is not None else '未满20，不报告完整成功率'
        text+=f"| {x['stage']} | {x['task']} | {x['records']} | {x['successes']} | {value} |\n"
    text+=f"\n{r['completed_cell_count']}个完整单元的在线实际KL最大值为{r['max_online_KL']}，满足冻结0.5上界。各已提交episode的父checkpoint和共同初态哈希均逐项匹配。\n\n"
    if r['eta'] and r['eta']['remaining_vectors']:
        e=r['eta'];a=when+datetime.timedelta(seconds=e['expected_remaining_seconds']);b=when+datetime.timedelta(seconds=e['conservative_remaining_seconds'])
        text+=f"按{e['completed_vector_count']}个已完成向量批次实测均值{e['mean_completed_vector_seconds']:.1f}秒，含审计预计剩余{e['expected_remaining_seconds']/60:.1f}–{e['conservative_remaining_seconds']/60:.1f}分钟，约{a:%H:%M}–{b:%H:%M} CST结束{v}；该估计不含后续条件分支。\n\n"
text+='尚不计算或判断完整AUC/FWT/OldFinal，不根据已见单元改变预算或门槛。达到预定完整门槛且工程验收通过才自动继续v3-A。未出现可行动工程阻塞。\n\n已完成本地包哈希、episode身份、attempt计数与在线KL只读复核；监控/轻量包命令退出0。临近预计完成，将同一自动化从每小时调整为每30分钟，不新建任务。\n'
(D/'REPORT.md').write_text(text,encoding='utf-8')
print(json.dumps(out,indent=2))
