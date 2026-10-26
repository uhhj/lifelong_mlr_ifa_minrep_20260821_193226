"""Recompute accepted historical Random+IFA baseline; no models or environments."""
from pathlib import Path
from fractions import Fraction as F
import datetime
import hashlib
import json

R = Path(__file__).resolve().parent
P = R.parent / '20261019_ra_coreset_goal_seed41/progress_20261019_222601'
D = R / 'final_readonly_20261026_014654'
sources = {}
def load(p):
    raw = p.read_bytes()
    sources[str(p)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def metrics(r):
    return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,
        FWT=sum(r[t,t] for t in range(6,10))/4,
        NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
        OldFinal=sum(r[9,t] for t in range(6,9))/3,
        FinalAll=sum(r[9,t] for t in range(6,10))/4)

old = load(P/'LOCAL_FINAL_ACCEPTANCE.json')
current = load(D/'LOCAL_FINAL_AUDIT.json')
assert old['status'] == 'PASS' and old['experiment_status'] == 'COMPLETED'
assert current['status'] == 'PASS_COMPLETE_LOCAL_ACCEPTANCE'
ev = load(D/'EVIDENCE.json')
records = json.loads(ev['light']['OUTCOMES.json']['text'])
rr = {}; bindings = []
for stage in range(6,10):
    for task in range(6,stage+1):
        p = P/f'completion_baseline/random/evaluation/trials_stage{stage}_task{task}.jsonl'
        raw = p.read_bytes(); sources[str(p)] = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(x) for x in raw.decode('utf8').splitlines() if x]
        assert len(rows) == 20
        assert sorted(x['initial_state_index'] for x in rows) == list(range(20))
        assert all(x['arm']=='random' and x['seed']==41 and x['stage']==stage and x['task']==task and type(x['success']) is bool for x in rows)
        rr[stage,task] = F(sum(x['success'] for x in rows),20)
        for row in rows:
            matched = [x for x in records if x['stage']==stage and x['task']==task and x['initial_index']==row['initial_state_index']]
            assert len(matched)==3
            assert all(x['initial_sha256']==row['initial_state_sha256'] and x['initial_file_sha256']==row['initial_file_sha256'] for x in matched)
        bindings.append(dict(stage=stage,task=task,records=20,initial_hashes_match=True,noise_pairing_verified=False))
bm = metrics(rr)
assert all(str(bm[k])==v['fraction'] for k,v in old['metrics']['random'].items())
ms = {'MLR_IFA':bm}; matrices = {'MLR_IFA':rr}
prior = load(R/'RA_ALIGN_BASELINE_COMPARISON.json')
matrices['RA_ALIGN'] = {tuple(map(int,k.split('_'))):F(str(v))/100 for k,v in prior['matrix']['RA_ALIGN'].items()}
ms['RA_ALIGN'] = metrics(matrices['RA_ALIGN'])
for arm in 'CDOS':
    ar = {tuple(map(int,k.split('_'))):F(v['successes'],20) for k,v in current['matrix'][arm].items()}
    matrices[arm] = ar; ms[arm] = metrics(ar)
    assert all(str(ms[arm][k])==v['fraction'] for k,v in current['metrics'][arm].items())
out = dict(status='PASS_HISTORICAL_DESCRIPTIVE_COMPARISON',observation=datetime.datetime.now().astimezone().isoformat(),
    baseline_definition='Project paper-aligned original MLR+IFA reproduction: Random+IFA, GOAL seed41; not paper-published scores, not Coreset/RA/ALIGN.',
    metrics={a:{k:dict(fraction=str(v),percent=float(v*100)) for k,v in m.items()} for a,m in ms.items()},
    delta_pp={a:{k:float((v-bm[k])*100) for k,v in m.items()} for a,m in ms.items()},
    matrix={a:{f'{s}_{t}':float(v*100) for (s,t),v in r.items()} for a,r in matrices.items()},
    negative_cells={a:{f'{s}_{t}':float((v-rr[s,t])*100) for (s,t),v in r.items() if v<rr[s,t]} for a,r in matrices.items()},
    initial_bindings=bindings,source_sha256=sources,additional_training=0,additional_forward=0,additional_environment=0,
    limits=['Historical baseline uses stage-global action RNG; DOS use per-trajectory paired Normal noise. Same initial hashes are not same action-noise streams.',
    'DOS include RA, ALIGN, C objectives and extra training interaction; total gain cannot be attributed only to O/S retention or success filtering.',
    'Original C comparator and all preregistered stopping decisions remain unchanged.',
    'Single development seed with 20 episodes/cell; no significance or independent-confirmation claim.',
    'Original reproduction uses state=True and completed schedule; official pretraining modality metadata remains unknown.'])
(R/'ORIGINAL_MLR_IFA_COMPARISON.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
lines=['# 原MLR+IFA基线对比：GOAL seed41','','本地复算已验收200条原Random+IFA回合；0新增训练、模型前向或环境。原MLR+IFA指项目按论文描述对齐的Random+IFA复现，不是论文原表成绩；未加Coreset、RA、ALIGN或CCBP辅助项。','','|方法|AUC|ΔAUC pp|FWT|NBT↓|OldFinal|FinalAll|ΔFinalAll pp|','|---|---:|---:|---:|---:|---:|---:|---:|']
for arm,m in ms.items():
    lines.append('|'+arm+'|'+'|'.join(f'{x:.4f}' for x in [float(m['AUC']*100),out['delta_pp'][arm]['AUC'],float(m['FWT']*100),float(m['NBT']*100),float(m['OldFinal']*100),float(m['FinalAll']*100),out['delta_pp'][arm]['FinalAll']])+'|')
lines += ['','|方法|最终Task6|Task7|Task8|Task9|','|---|---:|---:|---:|---:|']
for arm,r in matrices.items():
    lines.append('|'+arm+'|'+'|'.join(f'{float(r[9,t]*100):.1f}' for t in range(6,10))+'|')
lines += ['','完整三角矩阵与所有低于基线的单元见JSON。AUC按每任务已见阶段平均再跨任务平均；NBT按本项目定义越低越好，不能把更高最终成功率等同于更低遗忘。',
    '', '原基线和DOS的200/600条对应初态文件及逐状态SHA一致；原基线每stage重置全局动作RNG，DOS按轨迹独立噪声，因此不是完全配对的同随机流消融。各组主训练均400epoch/76100步，但DOS额外采集与保持计算另计。',
    '', '相对原MLR+IFA的总提升包含RA选样、ALIGN监督、C专家/路由辅助项以及D/O/S保持等多项变化，不能全部归因于新保持模块。仍保留原C强对照及未过整体门的结论。单seed开发数据不作显著性或独立确认声称。', '', *out['limits']]
(R/'ORIGINAL_MLR_IFA_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps({k:out[k] for k in ['metrics','delta_pp','negative_cells','matrix']},indent=2))
