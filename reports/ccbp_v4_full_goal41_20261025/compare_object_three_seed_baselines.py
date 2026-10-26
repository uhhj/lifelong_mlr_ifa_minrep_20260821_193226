"""Read-only nine-run comparison: original MLR+IFA, RA, RA+ALIGN, OBJECT41-43."""
from pathlib import Path
from fractions import Fraction as F
import json, hashlib, statistics, datetime

R=Path(__file__).resolve().parent
E=R.parent
sources={}
def read(p):
    b=p.read_bytes();sources[str(p)]=hashlib.sha256(b).hexdigest()
    return b.decode('utf8')
def js(p):return json.loads(read(p))
def metric(r):
    return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,
        FWT=sum(r[t,t] for t in range(6,10))/4,
        NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
        OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def trials(folder,seed,arm):
    rates={};identity={}
    for s in range(6,10):
        for t in range(6,s+1):
            rows=[json.loads(x) for x in read(folder/f'trials_stage{s}_task{t}.jsonl').splitlines() if x]
            assert len(rows)==20 and sorted(x['initial_state_index'] for x in rows)==list(range(20))
            assert all(x['seed']==seed and x['arm']==arm and x['stage']==s and x['task']==t and type(x['success']) is bool for x in rows)
            rates[s,t]=F(sum(x['success'] for x in rows),20)
            for x in rows:identity[s,t,x['initial_state_index']]=(x['initial_file_sha256'],x['initial_state_sha256'])
    return rates,identity

results=[];identities={};numeric={};matrices={};deltas=[]
historical=js(E/'20261020_all_methods_summary/ALL_RESULTS.json')['runs']
read(E/'20261019_ra_object_conditional/PROTOCOL.md')
for seed,stamp in [(41,'220252'),(42,'012252'),(43,'032427')]:
    if seed==41:
        p=E/'20261022_caet_v3_tac_eftr_object41/final_ALIGN_seed41_20261022_220252'
        b=E/'20261016_seeds40_41_three_arm/completion_evidence/seed41/rc'
    elif seed==42:
        p=E/f'20261022_align_object_extension_goal_conditional/final_OBJECT_{seed}_20261023_{stamp}'
        b=E/'20261013_coreset_extension/final_evidence/remote/seed42'
    else:
        p=E/f'20261022_align_object_extension_goal_conditional/final_OBJECT_{seed}_20261023_{stamp}'
        b=E/'20261016_seed43_three_arm_repeat_v1/completion_evidence/RC'
    accepted=js(p/'LOCAL_FINAL_AUDIT.json')
    assert accepted['status']=='COMPLETED_LOCAL_INDEPENDENT_PASS'
    ba=js(b/'INDEPENDENT_ACCEPTANCE.json');assert ba['status']=='PASS'
    version='重跑repeat1' if seed==43 else '首次'
    match=[x for x in historical if x['suite']=='OBJECT' and x['method']=='Random＋IFA' and x['seed']==seed and x['version']==version]
    assert len(match)==1
    bm={k:{'fraction':v} for k,v in match[0]['exact'].items()};bf=b/'random/evaluation'
    configs=[('MLR_IFA',bf,'random',bm),('RA',p/'baseline/ra_coreset/evaluation','ra_coreset',accepted['ra' if seed==41 else 'baseline']),('RA_ALIGN',p/'current/ra_coreset/evaluation','ra_coreset',accepted['metrics'])]
    for name,folder,arm,expected in configs:
        r,ids=trials(folder,seed,arm);m=metric(r)
        assert all(m[k]==F(v['fraction']) for k,v in expected.items())
        if name in ('RA','RA_ALIGN'):
            key=('ra' if seed==41 else 'parent') if name=='RA' else 'current'
            assert all(r[x['stage'],x['task']]==F(x[key]) for x in accepted['matrix'])
        identities[seed,name]=ids;numeric[seed,name]=m;matrices[seed,name]=r
        results.append(dict(seed=seed,method=name,records=200,metrics={k:dict(fraction=str(v),percent=float(v*100)) for k,v in m.items()},matrix={f'{s}_{t}':float(v*100) for (s,t),v in r.items()}))
    assert identities[seed,'MLR_IFA']==identities[seed,'RA']==identities[seed,'RA_ALIGN']
    for new,base in [('RA','MLR_IFA'),('RA_ALIGN','MLR_IFA'),('RA_ALIGN','RA')]:
        deltas.append(dict(seed=seed,comparison=f'{new}-{base}',delta_pp={k:float((numeric[seed,new][k]-numeric[seed,base][k])*100) for k in numeric[seed,new]},negative_cells={f'{s}_{t}':float((v-matrices[seed,base][s,t])*100) for (s,t),v in matrices[seed,new].items() if v<matrices[seed,base][s,t]}))
# Preserve first seed43 as a separate sensitivity analysis, never silently switch it.
first=E/'20261013_coreset_extension/final_evidence/remote/seed43'
assert js(first/'INDEPENDENT_ACCEPTANCE.json')['status']=='PASS'
fr,fi=trials(first/'random/evaluation',43,'random');fm=metric(fr)
assert all(fm[k]==F(v['fraction']) for k,v in js(first/'SUMMARY.json')['arms']['random']['metrics'].items())
sensitivity=dict(first_seed43={k:float(v*100) for k,v in fm.items()},
    original_mean_using_first={k:float((numeric[41,'MLR_IFA'][k]+numeric[42,'MLR_IFA'][k]+fm[k])/3*100) for k in fm})
summary={}
for name in ['MLR_IFA','RA','RA_ALIGN']:
    summary[name]={k:dict(mean_percent=float(sum(numeric[s,name][k] for s in (41,42,43))/3*100),sample_sd_pp=statistics.stdev(float(numeric[s,name][k]*100) for s in (41,42,43))) for k in numeric[41,name]}
out=dict(status='PASS_LOCAL_READONLY_RECOMPUTATION',time=datetime.datetime.now().astimezone().isoformat(),suite='OBJECT',seed43_baseline='repeat1 per frozen RA protocol',sensitivity=sensitivity,seeds=[41,42,43],runs=results,summary=summary,deltas=deltas,source_sha256=sources,formal_records_checked=2000,initial_hashes_equal_within_seed=True,action_noise_pairing_verified=False,new_training=0,new_forward=0,new_environment=0,
    definitions={'MLR_IFA':'Project original paper-aligned Random+IFA reproduction; not paper table scores.','RA':'RA-Coreset+IFA, no ALIGN.','RA_ALIGN':'RA-Coreset+IFA with ALIGN; not expert-auxiliary C.'},
    limits=['Historical stage-global action RNG does not establish identical action noise for trajectories across methods.','Three development seeds, 20 episodes/cell; means are descriptive, no significance claim.','Original replication state=True and schedule completion; official pretraining modality metadata unknown.','Main original baseline uses seed43 repeat1 per frozen RA protocol; first seed43 is separate sensitivity, not a fourth seed.'])
(R/'OBJECT_THREE_SEED_BASELINES.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
lines=['# OBJECT seed41/42/43：原MLR+IFA、RA、RA+ALIGN','','主表1800条已完成回合及seed43首次敏感性200条逐条复算；本轮0新训练/前向/环境。原MLR+IFA为Random+IFA复现；RA为RA-Coreset+IFA；RA+ALIGN不含C组额外专家监督。','','|seed|方法|AUC|FWT|NBT↓|OldFinal|FinalAll|','|---|---|---:|---:|---:|---:|---:|']
for x in results:lines.append('|'+str(x['seed'])+'|'+x['method']+'|'+'|'.join(f'{v["percent"]:.4f}' for v in x['metrics'].values())+'|')
lines+=['','三seed等权均值±样本SD（ddof=1）：','','|方法|AUC|FWT|NBT↓|OldFinal|FinalAll|','|---|---:|---:|---:|---:|---:|']
for name,m in summary.items():lines.append('|'+name+'|'+'|'.join(f'{v["mean_percent"]:.4f} ± {v["sample_sd_pp"]:.4f}' for v in m.values())+'|')
lines+=['','|seed|方法|最终Task6|Task7|Task8|Task9|','|---|---|---:|---:|---:|---:|']
for x in results:lines.append('|'+str(x['seed'])+'|'+x['method']+'|'+'|'.join(str(x['matrix'][f'9_{t}']) for t in range(6,10))+'|')
lines+=['','主表seed43原MLR+IFA采用冻结RA协议指定的repeat1；首次另作敏感性，不按结果选择。ALIGN相对RA三个seed AUC均提高，但最终旧任务平均在seed41/43下降，seed43最终四任务平均也下降。',
    '', 'AUC先对各任务已见阶段平均再跨任务平均；FWT沿用项目对角线平均定义；NBT是相对刚学完的后续平均下降，越低越好；OldFinal/FinalAll为最后阶段旧3任务/全部4任务平均。NBT不能替代绝对成功率。',
    '', '同seed三组200条对应初态文件/逐状态SHA一致；未声称动作噪声逐轨迹完全配对。完整矩阵、负单元、源SHA和精确分数见JSON。', '',*out['limits']]
lines += ['', 'seed43首次原MLR+IFA敏感性：'+json.dumps(sensitivity,ensure_ascii=False), '', '未更改任何实验合同或重新训练/评估。原MLR+IFA是项目复现成绩，不是论文表格原值。']
(R/'OBJECT_THREE_SEED_BASELINES.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(summary=summary,sensitivity=sensitivity,runs=[{k:x[k] for k in ['seed','method','metrics','matrix']} for x in results],deltas=deltas),indent=2))
