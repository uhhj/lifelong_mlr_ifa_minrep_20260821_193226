"""Read-only multi-stage progress accounting; no model or environment execution."""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def qualifies(base,child):
    if any(r['success'] is None for r in base+child):return False
    return (sum(not a['success'] and b['success'] for a,b in zip(base,child))>=1
            and not any(a['success'] and not b['success'] for a,b in zip(base,child)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args()
    p=args.directory.resolve();assert p.parent==R
    snap=read(p/'SNAPSHOT.json');d=read(p/'DETAILS.json');run=snap['runs']['RCTR_OBJECT_42']
    assert d['status']=='FROZEN_CONTRACT_INPUTS_PASS' and d['source_files']==176
    assert snap['controller_v2']['exit'] is None
    assert len(read(p/'REGISTRY.json'))==18
    assert snap['libero50_acquisition']['verified']==50 and snap['libero50_acquisition']['exit']=='0'
    history=d['epochs'];batch={6:157,7:227,8:309,9:362};steps=0;seen=set()
    for row in history:
        key=(row['task'],row['epoch']);assert key not in seen;seen.add(key)
        assert row['batches']==batch[row['task']]
        steps+=row['batches'];assert row['steps']==steps
    for t in sorted({x['task'] for x in history}):
        epochs=[x['epoch'] for x in history if x['task']==t]
        assert epochs==list(range(len(epochs))) and len(epochs)<=100
    assert len(history)>=run['epochs'] and steps>=run['optimizer_steps']
    assert all(v=='0' for v in d['exits'].values()),d['exits']
    summaries={}
    for task,data in d['collections_by_task'].items():
        t=int(task);rows=data['committed'];assert t in (6,7,8)
        assert len({r['key'] for r in rows})==len(rows)
        for r in rows:
            assert r['task']==t and 20<=r['initial']<=25 and r['snapshot_step'] in (40,120)
            assert r['phase'] in ('exploration','confirmation') and r['repeat'] in (0,1)
            assert 0<=r['steps']<=600-r['snapshot_step']
            assert r['status'] in ('COMMITTED','SKIPPED')
            assert isinstance(r['success'],bool) if r['status']=='COMMITTED' else r['success'] is None
            seed=100000000*42+1000000*t+1000*r['initial']+2*r['snapshot_step']+r['repeat']
            assert r['noise_seed']==seed+(500000 if r['phase']=='confirmation' else 0)
        phase=collections.Counter(r['phase'] for r in rows)
        assert phase['exploration']<=456 and phase['confirmation']<=48
        starts=collections.Counter(data['branch_starts']);ends=collections.Counter(data['branch_completes'])
        assert all(v==1 for v in ends.values())
        assert set(ends)<=set(starts) and {r['key'] for r in rows}<=set(starts)
        acceptance=data['acceptance'];certifications=[]
        if acceptance:
            assert acceptance['status']=='PASS' and acceptance['unique_branches']==len(rows)
            assert acceptance['sidecar_sha256']==data['sidecar_sha256']
            assert acceptance['contract_sha256']==d['contract_sha256']
            assert acceptance['invocation_counts']['screening_steps']==sum(r['steps'] for r in rows)
            assert acceptance['all_attempts']==sum(data['events'].values())
            assert all(v==1 for v in starts.values()) and set(starts)==set(ends)=={r['key'] for r in rows}
            for initial,at in sorted({(r['initial'],r['snapshot_step']) for r in rows}):
                table={(r['phase'],r['candidate']['id'],r['repeat']):r for r in rows
                       if (r['initial'],r['snapshot_step'])==(initial,at)}
                assert sum(k[0]=='exploration' for k in table)==38
                candidates=[]
                for axis in range(3):
                    for sign in (-1,1):
                        base=[table['exploration',f'a{axis}_{sign:+d}_original',rep] for rep in range(2)]
                        for role in ('1','2'):
                            cid=f'a{axis}_{sign:+d}_{role}'
                            child=[table['exploration',cid,rep] for rep in range(2)]
                            if qualifies(base,child):candidates.append((cid,axis,sign))
                certified=False
                if candidates:
                    cid,axis,sign=candidates[0]
                    base=[table['confirmation',f'a{axis}_{sign:+d}_original',rep] for rep in range(2)]
                    child=[table['confirmation',cid,rep] for rep in range(2)]
                    assert sum(k[0]=='confirmation' for k in table)==4
                    certified=qualifies(base,child)
                else:assert not any(k[0]=='confirmation' for k in table)
                certifications.append(dict(initial=initial,snapshot_step=at,
                    exploratory_candidates=len(candidates),certified=certified))
            assert sum(x['certified'] for x in certifications)==acceptance['records']
            assert acceptance['optimizer_calls']==0
        summaries[task]=dict(branches=len(rows),exploration=phase['exploration'],confirmation=phase['confirmation'],
            policy_steps=sum(r['steps'] for r in rows),successful_branches=sum(r['success'] is True for r in rows),
            accepted=bool(acceptance),certified_records=acceptance['records'] if acceptance else None,
            repeated_start_keys=sum(v>1 for v in starts.values()),
            pending_branch_keys=sorted(set(starts)-set(ends)),certifications=certifications,
            attempt_event_counts=data['events'])
    for task,aux in d['auxiliary'].items():
        t=int(task);h=[r for r in history if r['task']==t]
        assert aux['epoch_files']==len(h) and aux['updates']==sum(r['batches'] for r in h)
        assert aux['all_steps']==list(range(aux['first_step'],aux['last_step']+1))
        old=sum(summaries[str(j)]['certified_records'] or 0 for j in range(6,t))
        assert aux['active_updates']==(aux['updates'] if old else 0)
    at=datetime.datetime.fromtimestamp(d['unix'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    phase=d['run_status']['phase'];active=int(phase[-1]) if phase.startswith('collect') else None
    estimate=None
    if active:
        rr=d['collections_by_task'][str(active)]['committed']
        if rr:
            seconds=sum(r['seconds'] for r in rr)/len(rr)
            count=summaries[str(active)]
            estimate=dict(task=active,observed_branches=len(rr),seconds_per_branch=seconds,
                remaining_upper_hours=((456-count['exploration'])+(48-count['confirmation']))*seconds/3600,
                provisional=True,basis='Current-stage branch execution times; early successes may understate future cost')
    result=dict(status='PARTIAL_PROGRESS_AUDIT_PASS',observed_at=at,phase=phase,
        epochs=len(history),optimizer_steps=steps,formal_records=run['formal_records'],
        collections=summaries,auxiliary={t:{k:v for k,v in a.items() if k not in ('all_steps','file_sha256')}
                                       for t,a in d['auxiliary'].items()},
        current_stage_eta=estimate,source_and_input_hashes_pass=True,source_files=176,
        command_exits=d['exits'],libero50_gpu_admitted=False,experiment_recovery_performed=False,
        files={f.name:sha(f) for f in p.iterdir() if f.is_file() and f.name not in ('AUDIT.json','REPORT.md')})
    (p/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    report=f'# 阶段交接只读核查 — PARTIAL\n\n观察时间：{at}。当前 {phase}，累计{len(history)} epoch、{steps}次主更新，正式评估{run["formal_records"]}条。\n\n'
    for t,s in summaries.items():
        report+=f'- Task{t}：{s["branches"]}个辅助分支（探索{s["exploration"]}、确认{s["confirmation"]}），{s["policy_steps"]}策略步；完成采集验收={s["accepted"]}；认证训练记录={s["certified_records"]}。\n'
    report+='\n辅助分支成功不等于通过两级认证。已完成阶段按原始逐分支结果独立复算候选排序、探索与确认门槛；未放宽阈值或追加预算。\n\n'
    for t,a in result['auxiliary'].items():
        report+=f'Task{t}训练：{a["updates"]}次更新，其中RCTR辅助激活{a["active_updates"]}次。\n\n'
    if estimate:report+=f'当前Task{active}仅{estimate["observed_branches"]}分支，实测均值{estimate["seconds_per_branch"]:.2f}秒/分支，若持续该速度，剩余探索与确认上限约{estimate["remaining_upper_hours"]:.2f}小时。早期成功较快，阶段初期估时不稳，不能作为整套实验ETA。\n\n'
    report+='176个冻结源码及输入hash通过；本轮只更新本地外置读数与复算脚本，没有修改服务器科学代码、合同、模型或运行流程。全部实际已产生阶段退出码为0，队列尚未退出。LIBERO50数据就绪但GPU尚未准入。\n'
    (p/'REPORT.md').write_text(report,encoding='utf8')
    state=read(R/'RUN_STATUS.json');state.update(status='PARTIAL',observed_at=at,phase=phase,
        epochs=len(history),optimizer_steps=steps,formal_records=run['formal_records'],
        collection_branches=sum(x['branches'] for x in summaries.values()),collections=summaries,
        latest_progress=p.name,current_stage_eta=estimate)
    state.pop('seconds_per_collection_branch',None)
    (R/'RUN_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
