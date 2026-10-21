"""Metadata-only independent final acceptance. Refuses partial or still-running CAET."""
import argparse
import datetime
import json
from pathlib import Path
from fractions import Fraction as F
from audit_caet_progress import audit_formal, read, sha, R
from audit_rctr_completion_caet_progress import metrics, pack

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path)
    ap.add_argument('--seed',type=int,choices=(41,42,43),default=42)
    args=ap.parse_args();seed=args.seed
    p=args.directory.resolve();assert p.parent==R
    details_name='CAET_DETAILS.json' if seed==42 else f'CAET_DETAILS_seed{seed}.json'
    suffix='' if seed==42 else f'_seed{seed}'
    d=read(p/details_name);c=d['caet'];complete=c['completion'];a=complete['audit']
    assert f'CAET_OBJECT_seed{seed}_' in c['output']
    assert d['status']=='CAET_FROZEN_CONTRACT_INPUTS_PASS' and c['source_input_feature_hashes_pass']
    assert not c['workers']
    assert c['phase_exits']['pipeline.exit']==c['phase_exits']['supervisor.exit']=='0'
    assert all(x=='0' for x in c['phase_exits'].values())
    assert a['status']==complete['process_exit']['status']=='PASS'
    assert complete['process_exit']['pipeline_exit']==0 and not complete['process_exit']['residual_workers']
    assert complete['process_exit']['audit_sha256']==complete['audit_sha256']
    assert c['initial_rng_equal'] and c['task6_parity']['equal']
    assert c['launch']['status']==c['preflight']['status']=='PASS'
    assert c['launch']['contract_sha256']==c['contract_sha256']
    assert c['preflight']['optimizer_calls']==4 and c['preflight']['environment_steps']==0
    h=c['epochs'];assert len(h)==400
    assert {(r['task'],r['epoch']) for r in h}=={(t,e) for t in range(6,10) for e in range(100)}
    steps=0
    for row in h:
        assert row['batches']=={6:157,7:227,8:309,9:362}[row['task']]
        steps+=row['batches'];assert row['steps']==steps
    assert steps==105500 and a['epochs']==400 and a['optimizer_steps']==105500
    assert len(a['checkpoints'])==4
    for cp in a['checkpoints']:
        assert cp['strict_loaded_keys']==100
        assert complete['checkpoints_sha256'][f"mlrifa_{cp['task']}_model_ep99.pth"]==cp['sha256']
    for t,q in zip(range(6,10),(781,740,780,660)):
        for s in (c['storage_metadata'][str(t)],a['selection'][str(t)]):
            assert s['count']==q and s['ordered_ids_equal'] and s['payloads_equal']
        assert a['ifa'][str(t)]['equal'] and c['ifa_equal'][f'ifa_task{t}.json']
        x=c['auxiliary'][str(t)]
        assert x['epoch_files']==100 and x['updates']=={6:15700,7:22700,8:30900,9:36200}[t]
        assert 0<=x['active_updates']<=x['updates']
    assert c['auxiliary']['6']['active_updates']==0 and a['all_inputs_unchanged']
    assert len(a['auxiliary']['labels'])==4 and len(a['auxiliary']['predictors'])==3
    assert all(x['physical_recomputation_exact'] for x in a['auxiliary']['labels'])
    for x in a['auxiliary']['predictors']:
        assert x['status']=='PASS' and x['epochs']==10 and x['policy_updates']==0
        assert x['observed_training_positive_pairs']>0 and x['action_gradient_finite'] and x['action_gradient_l1']>0
        assert x['optimizer_steps']==10*((x['train_windows']+63)//64)
    fa=audit_formal(c['formal'],seed)
    assert fa['records']==200 and fa['completed_vectors']==fa['started_vectors']==20 and fa['completed_cells']==10
    rates,m,new=metrics(c['formal']['trials'],seed);base,b,old=metrics(complete['baseline_trials'],seed)
    assert all(r['seed']==seed for r in complete['baseline_trials'])
    for key,r in new.items():
        assert r['initial_state_sha256']==old[key]['initial_state_sha256'] and r['modal_sha256']==old[key]['modal_sha256']
        assert r['checkpoint_sha256']==complete['checkpoints_sha256'][f"mlrifa_{r['stage']}_model_ep99.pth"]
    assert all(F(a['metrics'][k]['fraction'])==v for k,v in m.items())
    assert all(F(a['baseline'][k]['fraction'])==v for k,v in b.items())
    gate=m['AUC']>b['AUC'];assert gate==a['metric_gate']
    receipt=complete['receipt']
    assert receipt['engineering']=='PASS' and not receipt['residual_workers']
    assert receipt['method']=='CAET' and receipt['suite']=='OBJECT' and receipt['seed']==seed
    assert receipt['formal_records']==receipt['baseline_records']==200
    assert F(receipt['auc'])==m['AUC'] and F(receipt['baseline_auc'])==b['AUC']
    result=dict(status='COMPLETED_GAIN' if gate else 'COMPLETED_NO_GAIN',seed=seed,engineering='LOCAL_INDEPENDENT_PASS',
        observed_at=datetime.datetime.fromtimestamp(d['observed_end_unix']).astimezone().isoformat(),
        metrics=pack(m),baseline=pack(b),delta=pack({k:m[k]-b[k] for k in m}),gate=gate,
        matrix=[dict(stage=s,task=t,current=str(rates[s,t]),baseline=str(base[s,t])) for s,t in rates],
        formal=fa,epochs=400,optimizer_steps=steps,checkpoints=a['checkpoints'],
        auxiliary=c['auxiliary'],effect_audit=a['auxiliary'],summed_epoch_seconds=a['summed_epoch_seconds'],
        source_details_sha256=sha(p/details_name),server_audit_sha256=complete['audit_sha256'])
    (p/f'CAET_FINAL_AUDIT{suffix}.json').write_text(json.dumps(result,indent=2)+'\n')
    report=f'# CAET OBJECT seed{seed}最终独立复算\n\n状态：'+result['status']+'。工程验收全部通过。\n\n|指标|RA %|CAET %|差 pp|\n|---|---:|---:|---:|\n'
    for k in m:report+=f'|{k}|{float(100*b[k]):.6f}|{float(100*m[k]):.6f}|{float(100*(m[k]-b[k])):+.6f}|\n'
    report+='\n|阶段|任务|RA %|CAET %|\n|---|---|---:|---:|\n'
    for s,t in rates:report+=f'|{s}|{t}|{float(100*base[s,t]):.1f}|{float(100*rates[s,t]):.1f}|\n'
    report+='\n严格按精确AUC与工程验收判定扩展，不添加其他数值门槛。额外物理标签和效果拟合成本、单seed探索性、全局动作随机流未隔离、预训练state未知及调度补全边界均保留。辅助梯度实际激活不等同于机制的独立因果证明。\n'
    (p/f'CAET_FINAL_REPORT{suffix}.md').write_text(report,encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
