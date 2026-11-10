"""Read only accepted outcome records and source; no training, inference or rollouts."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,statistics,datetime
R=Path(__file__).resolve().parent
J=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=J(R/'ACCEPTANCE.json');D=R/a['final_directory'];final=J(D/'LOCAL_FINAL_AUDIT.json')
assert a['local_final_audit_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json')
old=J(R/'BASELINE_OUTCOMES.json')['cases'];bindings=J(R/'BASELINE_BINDING.json')['cases'];inputs=J(R/'INPUT_AUDIT.json')
percase={};suites={}
def matrix(records):
    assert len(records)==200
    result={}
    for s in range(6,10):
        for t in range(6,s+1):
            xs=[r for r in records if (r['stage'],r['task'])==(s,t)]
            assert sorted(r['initial_state_index'] for r in xs)==list(range(20))
            result[s,t]=F(sum(x['success'] for x in xs),20)*100
    return result
for case,new in final['cases'].items():
    before=matrix(old[case]);after=matrix(new['records'])
    assert all(float(v)==new['matrix'][f'{s}_{t}'] for (s,t),v in after.items())
    assert all(float(v)==bindings[case]['baseline_matrix'][f'{s}_{t}'] for (s,t),v in before.items())
    contribution={t:(sum(after[s,t]-before[s,t] for s in range(t,10))/(10-t))/4 for t in range(6,10)}
    auc_delta=sum(contribution.values())
    assert abs(float(auc_delta)-(new['metrics']['AUC']['percent']-bindings[case]['baseline_metrics']['AUC']['percent']))<1e-9
    percase[case]=dict(auc_delta_pp=float(auc_delta),task_auc_contribution_pp={str(t):float(v) for t,v in contribution.items()},
       task6_at_learning=dict(old=float(before[6,6]),new=float(after[6,6])),
       task9=dict(old=float(before[9,9]),new=float(after[9,9])),
       old_hardware=bindings[case]['historical_gpu'],
       evaluation_implementation='legacy_full_window' if case in final['evaluation_cache_boundary']['legacy_full_window_cases'] else 'validated_single_frame_cache',
       bc_invalid_percent={t:100*(1-v['bc_valid']/v['population']) for t,v in inputs['cases'][case]['stages'].items()})
for suite in ['OBJECT','GOAL']:
    group=[percase[f'{suite}_{seed}'] for seed in [41,42,43]]
    cs={str(t):statistics.mean(x['task_auc_contribution_pp'][str(t)] for x in group) for t in range(6,10)}
    suites[suite]=dict(auc_delta_pp=sum(cs.values()),task_auc_contribution_pp=cs,
        task6_plus9_share_of_net_auc_delta=(cs['6']+cs['9'])/sum(cs.values()),
        task9_old_successes=sum(x['task9']['old']/5 for x in group),task9_new_successes=sum(x['task9']['new']/5 for x in group),task9_episodes_per_method=60)
sources=[R/'BASELINE_OUTCOMES.json',R/'BASELINE_BINDING.json',R/'INPUT_AUDIT.json',D/'LOCAL_FINAL_AUDIT.json',
 R/'cases/OBJECT_41/source/libero/lifelong/models/base_policy.py',R/'cases/OBJECT_41/source/libero/lifelong/models/policy_head.py',
 R/'cases/OBJECT_41/source/coreset_exp/align_support.py',R/'input_audit_readonly.py']
out=dict(status='PASS_RAW_OUTCOME_GAIN_DECOMPOSITION_NOT_CAUSAL_ATTRIBUTION',observed=datetime.datetime.now().astimezone().isoformat(),
 new_records_recomputed=1200,old_records_recomputed=1200,per_case=percase,suites=suites,
 source_sha256={str(p.relative_to(R)):sha(p) for p in sources},
 verified_training_change=['BC endpoint stored action a[u] is replaced by label a[u+1] for valid windows.',
 'BC invalid windows are masked and averaged over valid windows only; the complete original sample population remains for IFA.',
 'Thus this is a target-time and boundary-mask intervention together, not a separately isolated one-step-target-only experiment.'],
 findings=['Arithmetic gain is reproduced from all 1200 new and 1200 old records, not a swapped mean or omitted seed.',
 'OBJECT41/42 retain legacy full-window evaluation and still show large gains: cache conversion cannot be the sole explanation.',
 'Task6 already differs before replay selection affects later tasks; this does not by itself isolate ALIGN from hardware/execution.',
 'No read-only evidence here quantifies the causal share of target shift, masking, hardware or execution randomness.',
 'CPU acceptance confirms record/model consistency, not strict causal comparability or statistical significance.'],
 new_training=0,new_optimizer=0,new_policy_forward=0,new_environment=0)
write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
write(R/'GAIN_MAGNITUDE_AUDIT_20261110.json',out)
lines=['# ALIGN大幅提升的只读复核','',f"观察：{out['observed']}。未运行训练、模型前向或环境。",'',
 '## 数字是否算错','',
 '重新读取全部1200条新记录与1200条既有Random记录，每case检查10单元、各20个唯一初态，并重建成功率矩阵。得到的AUC差值与最终报告一致，不是漏seed、选取最佳轮或均值错算。',
 '','## AUC提升来自哪里','',
 '采用既定AUC定义：每个任务先对其学习后各阶段取平均，再对4个任务等权平均。Task9虽只在最终阶段评估，仍占AUC的1/4。下表为各任务对总AUC差值的贡献，单位百分点。',
 '','|套件|Task6|Task7|Task8|Task9|合计|','|---|---:|---:|---:|---:|---:|']
for suite,s in suites.items():lines.append('|'+suite+'|'+'|'.join(f"{s['task_auc_contribution_pp'][str(t)]:+.4f}" for t in range(6,10))+f"|{s['auc_delta_pp']:+.4f}|")
lines+=['',
 'OBJECT的Task9由35/60成功变为56/60，即58.3333%→93.3333%，单项贡献8.75pp；Task6学习后全过程均值38.75%→62.9167%，贡献6.0417pp。二者合计解释约82.24%的总AUC增量。这说明收益集中，不是每个任务平均提高18pp。',
 '','## 缓存是否是全部原因','',
 'OBJECT41/42一直使用原全窗评估，AUC仍分别增加18.5417/17.2917pp；OBJECT43采用缓存增加18.1250pp。因此缓存替换不能单独解释全部大幅提升。该事实也不能证明缓存完全没有影响，更不能排除历史硬件与执行随机性的影响。',
 '','## ALIGN实际改了什么','',
 '源码中原GMMHead.loss_fn读取已存动作窗口末项target[:,-1,:]；输入审计将它对应到原始示范a[u]。本轮aligned_bc改用已绑定原始示范的a[u+1]，并仅对合法窗口计算BC均值。OBJECT阶段人口约4.78%–5.88%的窗口、GOAL约6.18%–9.25%的窗口不再贡献BC；这些样本仍保留在原人口中参与IFA。',
 '因此ALIGN同时改变动作监督时刻与边界mask，并不是只给原损失加一个很小的系数。这种训练目标变化可能带来较大闭环差异，但“可能”是机制解释，不是这18pp的已验证归因。此处不据代码差异断言原论文有bug。',
 'Task6已经出现差异：OBJECT41为50%→40%、42为30%→50%、43为40%→65%；GOAL41为70%→80%、42为65%→70%、43为80%→75%。方向并不统一，不能把它描述成每阶段都立即提升。',
 '','## 现有证据能得出的结论','',
 '可以确认：本轮完整记录的均值提升真实存在，数据、模型与预算验收通过，且有明确任务来源。不能确认：全部增量均为ALIGN净收益，或者单独来自下一动作标签而不是mask，或者与历史执行条件无关。原基线使用3090/4090，本轮使用4080SUPER；同初态不等于跨运行相同闭环随机轨迹。S与Random的已查明评估差异，也不能直接拿来解释本轮Random对Random+ALIGN的全部差值。',
 '本轮Random+ALIGN的AUC高于既有RA+ALIGN，OBJECT为86.2847对80.1736，GOAL为83.8194对81.1111。这要求重新检查“RA与ALIGN单调叠加”的表述，不能为满足预期而挑选、重命名或删除现有结果。',
 '用户明确不重跑：本次仅拆解现有证据，所有原结果、失败及执行边界保留。最终CPU接受表示保存结果通过一致性验收，不等于受控单因素因果证明。']
(R/'GAIN_MAGNITUDE_AUDIT_20261110.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],suites=suites),ensure_ascii=False))
