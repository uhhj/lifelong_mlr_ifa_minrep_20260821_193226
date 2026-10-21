"""Read-only local CAET progress audit; completed RCTR evidence is referenced, not rerun."""
import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def audit_formal(formal):
    trials=formal['trials'];events=formal['attempts']
    keys=[(r['stage'],r['task'],r['trial']) for r in trials]
    assert len(keys)==len(set(keys))<=200
    for r in trials:
        assert 6<=r['task']<=r['stage']<=9 and r['epoch']==99 and r['seed']==42
        assert 0<=r['trial']<20 and r['initial_state_index']==r['trial']
        assert r['initial_state_sha256']==r['initial_hashes'][r['trial']]
        assert isinstance(r['success'],bool) and 0<r['vector_group_steps']<=600
    assert not any(e['event']=='technical_exception' for e in events)
    starts={(e['stage'],e['task'],e['group']):e for e in events if e['event']=='vector_attempt_start'}
    done={(e['stage'],e['task'],e['group']):e for e in events if e['event']=='vector_attempt_complete'}
    assert len(starts)==sum(e['event']=='vector_attempt_start' for e in events)
    assert len(done)==sum(e['event']=='vector_attempt_complete' for e in events)
    assert set(done)<=set(starts) and len(starts)<=20 and len(starts)-len(done)<=1
    for k,e in starts.items():
        assert e['group'] in (0,1) and e['env_seed']==42+e['group']
        assert e['initial_indices']==list(range(10*e['group'],10*(e['group']+1)))
    for k,e in done.items():
        rows=sorted((r for r in trials if (r['stage'],r['task'],r['trial']//10)==k),key=lambda r:r['trial'])
        assert len(rows)==10 and [r['success'] for r in rows]==e['successes']
        assert all(r['vector_group_steps']==e['steps'] for r in rows)
        assert 0<e['steps']<=600 and e['lanes']==10
    for name,cell in formal['cells'].items():
        m=re.fullmatch(r'cell_stage(\d+)_task(\d+)\.json',name);assert m
        stage,task=map(int,m.groups())
        assert cell['stage']==stage and cell['epoch']==99
        assert Path(cell['trial_log']).name==f'trials_stage{stage}_task{task}.jsonl'
        assert Path(cell['attempt_log']).name==f'attempts_stage{stage}_task{task}.jsonl'
        rows=[r for r in trials if (r['stage'],r['task'])==(stage,task)]
        assert len(rows)==20 and sum(r['success'] for r in rows)/20==cell['success_rate']
        assert all(r['checkpoint_sha256']==cell['checkpoint_sha256'] and r['initial_hashes']==cell['initial_hashes'] for r in rows)
    durations=[e['unix']-starts[k]['unix'] for k,e in done.items()]
    avg=sum(durations)/len(durations) if durations else None
    return dict(records=len(trials),completed_vectors=len(done),started_vectors=len(starts),
        completed_cells=len(formal['cells']),technical_exceptions=0,
        mean_vector_seconds=avg,remaining_evaluation_hours=(20-len(done))*avg/3600 if avg else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path)
    p=ap.parse_args().directory.resolve();assert p.parent==R
    d=read(p/'CAET_DETAILS.json');c=d['caet'];snap=read(p/'SNAPSHOT.json')
    assert d['status']=='CAET_FROZEN_CONTRACT_INPUTS_PASS' and c['source_input_feature_hashes_pass']
    assert c['launch']['status']==c['preflight']['status']=='PASS'
    assert c['launch']['contract_sha256']==c['contract_sha256']
    assert c['preflight']['optimizer_calls']==4 and c['preflight']['environment_steps']==0
    assert c['task6_parity']['equal']
    if 'storage_metadata' in c and 'initial_rng_equal' in c:
        assert c['initial_rng_equal'] and all(c['ifa_equal'].values())
        for task,storage in c['storage_metadata'].items():
            assert storage['ordered_ids_equal'] and storage['payloads_equal']
            assert storage['count']=={6:781,7:740,8:780,9:660}[int(task)]
    assert all(v=='0' for v in c['admission_exits'].values())
    assert all(v=='0' for v in c['phase_exits'].values())
    history=c['epochs'];assert len({(r['task'],r['epoch']) for r in history})==len(history)
    steps=0
    for r in history:
        steps+=r['batches'];assert r['steps']==steps
    for t in range(6,10):
        rows=[r for r in history if r['task']==t]
        assert [r['epoch'] for r in rows]==list(range(len(rows))) and len(rows)<=100
        assert all(r['batches']=={6:157,7:227,8:309,9:362}[t] for r in rows)
        aux=c['auxiliary'][str(t)]
        assert aux['updates']==aux['epoch_files']*{6:157,7:227,8:309,9:362}[t]
        assert abs(aux['epoch_files']-len(rows))<=1  # Read during active atomic epoch commits.
        assert 0<=aux['active_updates']<=aux['updates']
    starts=[e['phase'] for e in c['command_events'] if e['event']=='start']
    assert len(starts)==len(set(starts)), 'Repeated phase start requires a separately audited recovery'
    assert c['auxiliary']['6']['active_updates']==0
    for effect in c['effects'].values():
        assert effect['status']=='PASS' and effect['epochs']==10 and effect['policy_updates']==0
        assert effect['global_positive_pairs']>0 and effect['observed_training_positive_pairs']>0
        assert effect['action_gradient_finite'] and effect['action_gradient_l1']>0
    assert snap['controller_v2']['exit'] is None
    assert snap['decision']['method']=='CAET'
    assert snap['libero50_acquisition']['verified']==50 and snap['libero50_acquisition']['exit']=='0'
    assert c['workers'], 'No active worker: inspect actual exits before proceeding'
    formal=c['formal'];keys=[(r['stage'],r['task'],r['trial']) for r in formal['trials']]
    assert len(keys)==len(set(keys))<=200
    for r in formal['trials']:
        assert r['seed']==42 and r['epoch']==99 and 0<=r['trial']<20
    formal_audit=audit_formal(formal)
    # This progress audit is deliberately not a substitute for final episode/CP/exit acceptance.
    timed=[r for r in history if r['task']==history[-1]['task'] and r['task']>6]
    sec_per_step=sum(r['seconds'] for r in timed)/sum(r['batches'] for r in timed)
    remaining=(105500-steps)*sec_per_step/3600
    previous=R/'progress_20261021_094302/FINAL_RCTR_AND_CAET_AUDIT.json'
    eval_ref=read(previous)['caet']['eta']['eval_reference_hours']
    at=datetime.datetime.fromtimestamp(d['observed_end_unix'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    result=dict(status='PARTIAL',audit='CAET_PROGRESS_PASS',observed_at=at,epochs=len(history),optimizer_steps=steps,
        latest_epoch=history[-1],formal_records=len(keys),auxiliary=c['auxiliary'],effects=c['effects'],
        formal_audit=formal_audit,
        source_files=c['source_files'],source_input_feature_hashes_pass=True,phase_exits=c['phase_exits'],
        storage=c.get('storage_metadata',{}),ifa_equal=c.get('ifa_equal',{}),
        eta=dict(remaining_main_training_hours=remaining,full_evaluation_reference_hours=eval_ref,
            remaining_evaluation_hours=formal_audit['remaining_evaluation_hours'],
            excludes='Future labels, effect fitting, storage and final audit; no whole-program ETA'),
        completed_rctr_evidence=dict(path=str(previous.relative_to(R)),sha256=sha(previous),recomputed_this_check=False),
        details_sha256=sha(p/'CAET_DETAILS.json'),snapshot_sha256=sha(p/'SNAPSHOT.json'),
        optimizer_calls_in_this_check=0,environment_steps_in_this_check=0)
    (p/'CAET_PROGRESS_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    report=f'# CAET OBJECT seed42进度（PARTIAL）\n\n观察时间：{at}。\n\n已完成{len(history)}个完整epoch、{steps}主优化步；当前Task{history[-1]["task"]}已完成epoch{history[-1]["epoch"]}（从0计数）。正式评估{len(keys)}/200条。\n\n'
    report+='|任务|完整辅助账目epoch|主更新数|效果辅助项实际激活|\n|---|---:|---:|---:|\n'
    for t,a in c['auxiliary'].items():report+=f'|{t}|{a["epoch_files"]}|{a["updates"]}|{a["active_updates"]}|\n'
    report+=f'\n{c["source_files"]}个冻结源码及输入/特征哈希、Task6同父张量、已完成阶段退出码和效果模型非零动作梯度检查通过。未重复GPU预检，无新增训练或环境调用。效果辅助项激活仅证明训练干预存在，不代表成功率改善。\n\n'
    report+=f'按最近任务实测吞吐，剩余主训练约{remaining:.2f}小时；完整评估参考约{eval_ref:.2f}小时，另需标签、效果拟合、入库和验收。预估未包含后续条件分支或LIBERO50预训练。\n\n'
    if formal_audit['completed_vectors']:
        report+=f'本轮评估已完成{formal_audit["completed_vectors"]}/20个vector、{formal_audit["completed_cells"]}个完整单元，0技术异常，初态/seed/步数/逐条结果与attempt账目一致。按实测均值{formal_audit["mean_vector_seconds"]:.1f}秒/vector，剩余评估约{formal_audit["remaining_evaluation_hours"]:.2f}小时，最终验收另计；早期吞吐可能不代表后续任务。训练完成不代表整轮完成。\n\n'
    report+='RCTR的完成及NO_GAIN结论沿用已验收证据，本轮不重复复算。LIBERO50数据已齐，尚未GPU准入。控制器同步等待当前launcher，因此旧queue状态键不用于判断重复启动。\n\n单seed探索性；全局动作RNG未逐轨迹隔离；预训练state元数据未知；调度补全及额外效果拟合成本均保留披露。最终指标和门槛须完整评估与验收后判断。\n'
    (p/'REPORT.md').write_text(report,encoding='utf8')
    state=read(R/'RUN_STATUS.json');state.update(status='PARTIAL',active='CAET_OBJECT_42',phase=c['status']['status'],
        observed_at=at,epochs=len(history),optimizer_steps=steps,formal_records=len(keys),latest_progress=p.name,
        caet_progress=result,current_stage_eta=result['eta'])
    (R/'RUN_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
