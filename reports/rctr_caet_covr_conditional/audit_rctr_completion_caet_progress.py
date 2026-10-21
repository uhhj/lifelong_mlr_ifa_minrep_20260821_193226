"""Local metadata-only completion recomputation and CAET admission/progress audit."""
import argparse
import collections
import datetime
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess
import sys

R=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pack(m): return {k:dict(fraction=str(v),percent=float(100*v)) for k,v in m.items()}
def metrics(rows, seed=42):
    keys=[(r['stage'],r['task'],r['trial']) for r in rows]
    assert len(keys)==len(set(keys))==200
    rates={}
    for stage in range(6,10):
        for task in range(6,stage+1):
            cell=[r for r in rows if (r['stage'],r['task'])==(stage,task)]
            assert len(cell)==20 and {r['trial'] for r in cell}==set(range(20))
            assert all(r['seed']==seed and r['epoch']==99 and isinstance(r['success'],bool) for r in cell)
            rates[stage,task]=F(sum(r['success'] for r in cell),20)
    m=dict(AUC=sum(sum(rates[s,t] for s in range(t,10))/F(10-t) for t in range(6,10))/4,
           FWT=sum(rates[t,t] for t in range(6,10))/4,
           NBT=sum(sum(rates[t,t]-rates[s,t] for s in range(t+1,10))/F(9-t) for t in range(6,9))/3,
           OldFinal=sum(rates[9,t] for t in range(6,9))/3,
           FinalAll=sum(rates[9,t] for t in range(6,10))/4)
    return rates,m,{(r['stage'],r['task'],r['trial']):r for r in rows}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);p=ap.parse_args().directory.resolve()
    assert p.parent==R
    subprocess.run([sys.executable,str(R/'audit_progress_stages.py'),str(p)],check=True,stdout=subprocess.DEVNULL)
    d=read(p/'DETAILS.json');a=read(p/'AUDIT.json');c=d['completion'];server=c['AUDIT.json']
    assert a['epochs']==400 and a['optimizer_steps']==105500 and a['formal_records']==200
    assert a['formal_audit']['completed_vectors']==20 and a['formal_audit']['completed_cells']==10
    assert not d['worker_processes']
    assert d['exits']['pipeline.exit']==d['exits']['supervisor.exit']=='0'
    assert server['status']==c['PROCESS_EXIT_ACCEPTANCE.json']['status']=='PASS'
    assert c['PROCESS_EXIT_ACCEPTANCE.json']['audit_sha256']==c['audit_sha256']
    assert c['PROCESS_EXIT_ACCEPTANCE.json']['pipeline_exit']==0
    assert c['TASK6_TENSOR_PARITY.json']['equal'] and c['initial_rng_equal']
    assert len(server['checkpoints'])==4
    for cp in server['checkpoints']:
        assert cp['strict_loaded_keys']==100
        assert c['checkpoints_sha256'][f"mlrifa_{cp['task']}_model_ep99.pth"]==cp['sha256']
    assert all(v['ordered_ids_equal'] and v['payloads_equal'] for v in server['selection'].values())
    assert [server['selection'][str(t)]['count'] for t in range(6,10)]==[781,740,780,660]
    assert all(v['equal'] for v in server['ifa'].values()) and server['all_inputs_unchanged']
    assert all(v['active_updates']==0 for v in a['auxiliary'].values())
    assert all(v['certified_records']==0 for v in a['collections'].values())
    rates,m,new=metrics(d['formal']['trials']);base,b,old=metrics(c['baseline_trials'])
    for key in new:
        for field in ('initial_state_sha256','modal_sha256'):
            assert new[key][field]==old[key][field]
    assert all(F(server['metrics'][k]['fraction'])==v for k,v in m.items())
    assert all(F(server['baseline'][k]['fraction'])==v for k,v in b.items())
    receipt=c['receipt'];assert receipt['engineering']=='PASS' and not receipt['residual_workers']
    assert F(receipt['auc'])==m['AUC'] and F(receipt['baseline_auc'])==b['AUC']
    gate=m['AUC']>b['AUC'];assert gate==server['metric_gate']==False
    caet=d['caet'];assert caet['source_input_feature_hashes_pass']
    assert caet['launch']['status']==caet['preflight']['status']=='PASS'
    assert caet['launch']['contract_sha256']==caet['contract_sha256']
    assert caet['preflight']['optimizer_calls']==4 and caet['preflight']['environment_steps']==0
    assert caet['task6_parity']['equal']
    assert all(v=='0' for v in caet['admission_exits'].values())
    assert all(v=='0' for v in caet['phase_exits'].values())
    history=caet['epochs'];assert len({(r['task'],r['epoch']) for r in history})==len(history)
    assert sum(r['batches'] for r in history)==history[-1]['steps']
    for t in range(6,10):
        rows=[r for r in history if r['task']==t]
        assert [r['epoch'] for r in rows]==list(range(len(rows))) and len(rows)<=100
        assert all(r['batches']=={6:157,7:227,8:309,9:362}[t] for r in rows)
    starts=[e['phase'] for e in caet['command_events'] if e['event']=='start']
    assert len(starts)==len(set(starts))
    assert caet['auxiliary']['6']['active_updates']==0
    assert caet['auxiliary']['7']['active_updates']>0
    for key,effect in caet['effects'].items():
        assert effect['status']=='PASS' and effect['epochs']==10
        assert effect['global_positive_pairs']>0 and effect['observed_training_positive_pairs']>0
        assert effect['action_gradient_finite'] and effect['action_gradient_l1']>0
        assert effect['policy_updates']==0
    task7=[r for r in history if r['task']==7]
    sec_per_batch=sum(r['seconds'] for r in task7)/sum(r['batches'] for r in task7)
    remaining_train=(105500-history[-1]['steps'])*sec_per_batch/3600
    at=datetime.datetime.fromtimestamp(d['observed_end_unix'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    delta={k:m[k]-b[k] for k in m}
    result=dict(status='RCTR_COMPLETED_NO_GAIN_CAET_PARTIAL',observed_at=at,
        rctr=dict(engineering='LOCAL_INDEPENDENT_PASS',metrics=pack(m),baseline=pack(b),delta=pack(delta),
            matrix=[dict(stage=s,task=t,rctr=str(rates[s,t]),baseline=str(base[s,t])) for s,t in rates],
            same_episode_outcomes=sum(new[k]['success']==old[k]['success'] for k in new),
            gate=gate,certified_records=0,auxiliary_activated_steps=0,formal_records=200,vector_attempts=20,
            training_epochs=400,optimizer_steps=105500,checkpoint_strict_load_evidence=server['checkpoints'],
            cost=dict(summed_epoch_seconds=server['summed_epoch_seconds'],auxiliary_branches=1388,
                auxiliary_policy_steps=296583,collection_prefix_steps=2250,collection_engineering_steps=576),
            scientific_conclusion='No certified RCTR supervision was activated; scores are not evidence of RCTR mechanism benefit.'),
        caet=dict(engineering='PROGRESS_AND_ADMISSION_PASS',epochs=len(history),optimizer_steps=history[-1]['steps'],
            source_files=caet['source_files'],auxiliary=caet['auxiliary'],effects=caet['effects'],formal_records=caet['formal_records'],
            eta=dict(remaining_policy_training_hours=remaining_train,eval_reference_hours=a['formal_audit']['mean_seconds_per_completed_vector']*20/3600,
                provisional=True,excludes='Future labels/fitting/storage/audit; extrapolated from Task7 observed training rate')),
        details_sha256=sha(p/'DETAILS.json'),branch_rules_sha256=sha(R/'BRANCH_RULES.md'))
    (p/'FINAL_RCTR_AND_CAET_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    report=f'# RCTR seed42完成与CAET交接\n\n观察时间：{at}。整体队列仍PARTIAL。\n\nRCTR已完成400epoch/105500步、4CP严格加载、200正式记录/20vector，退出和本地独立验收通过。\n\n|指标|RA基线 %|RCTR %|差 pp|\n|---|---:|---:|---:|\n'
    for k in m:report+=f'|{k}|{float(100*b[k]):.6f}|{float(100*m[k]):.6f}|{float(100*delta[k]):+.6f}|\n'
    report+='\n|阶段|Task|RA成功率 %|RCTR成功率 %|\n|---|---|---:|---:|\n'
    for s,t in rates:report+=f'|{s}|{t}|{float(100*base[s,t]):.1f}|{float(100*rates[s,t]):.1f}|\n'
    report+=f'\n两组逐episode成功/失败结果相同：{result["rctr"]["same_episode_outcomes"]}/200。AUC未严格提高，按冻结规则不扩RCTR41/43及GOAL，已自动进入CAET42。\n\n'
    report+='RCTR共1368探索及20确认分支、296583策略步，3阶段均0认证，所有主更新辅助激活0。该轮未形成RCTR监督干预，不能据此证明该监督目标有效或有害。所有失败和零记录如实保留，无追加交互。\n\n'
    report+=f'CAET已完成{len(history)}epoch/{history[-1]["steps"]}步；真实4次隔离GPU更新恢复预检、0环境PASS，Task6实际张量同父，源码/输入/特征哈希PASS。Task7辅助项已实际激活{caet["auxiliary"]["7"]["active_updates"]}次；效果预测器的留出误差不是闭环结果。\n\n'
    report+=f'按当前Task7吞吐，剩余主训练约{remaining_train:.2f}小时；参考RCTR正式评估约{result["caet"]["eta"]["eval_reference_hours"]:.2f}小时，另需后续标签生成/预测器拟合/入库与验收，不承诺精确结束时刻。\n\n'
    report+='限制：单seed探索性、额外采集/拟合成本、全局动作随机流未逐轨迹隔离、预训练state元数据unknown及调度补全。LIBERO50仍未GPU准入；其共享预训练为新复现，不是官方发布权重。\n'
    (p/'COMPLETION_REPORT.md').write_text(report,encoding='utf8')
    state=read(R/'RUN_STATUS.json');state.update(active='CAET_OBJECT_42',phase=caet['status']['status'],
        observed_at=at,epochs=len(history),optimizer_steps=history[-1]['steps'],formal_records=caet['formal_records'],
        latest_progress=p.name,rctr_completed=result['rctr'],caet_progress=result['caet'],current_stage_eta=result['caet']['eta'])
    for key in ('collections','formal_audit','collection_branches'):state.pop(key,None)
    (R/'RUN_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
