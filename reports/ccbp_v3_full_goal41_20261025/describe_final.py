"""Aggregate accepted local scalar evidence only; no training, forwards or environments."""
import datetime
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
D = Path(sys.argv[1]).resolve()
def read(p):
    return json.loads(p.read_text(encoding='utf8'))
def save(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

a = read(D/'LOCAL_FINAL_AUDIT.json')
assert a['status'] == 'PASS_COMPLETE_LOCAL_ACCEPTANCE'
v = read(D/'EVIDENCE.json')
light = {k: json.loads(x['text']) for k,x in v['light'].items() if k.endswith('.json')}
offline = {}
common_windows = {}
for stage in range(7,10):
    identities = {}
    for arm in 'CRG':
        raw = read(D/('diagnostics/%s_%d.json'%(arm,stage)))
        identities[arm] = [(r['index'],r['task'],r['sample_sha256']) for r in raw]
    assert identities['C']==identities['R']==identities['G']
    common_windows[str(stage)] = dict(count=len(identities['C']),identity_order_payload_sha_exact=True)
fields = ['expert_exec','expert_best','regret','teacher_distance_exec','directional_exec',
          'projection_fraction','margin_route','weighted_anchor','auxiliary_loss']
for arm in 'CRG':
    for stage in range(7,10):
        rows = read(D/('RECOMPUTED_%s_%d.json'%(arm,stage)))['rows']
        for task in sorted({r['task'] for r in rows}):
            rr = [r for r in rows if r['task']==task]
            z = dict(count=len(rr), nonbest_count=sum(r['regret']>1e-12 for r in rr),
                     positive_route_count=sum(r['margin_route']>0 for r in rr),
                     projection_at_teacher=sum(r['projection_fraction']<=1e-10 for r in rr),
                     projection_at_expert=sum(r['projection_fraction']>=1-1e-10 for r in rr))
            z['statistics'] = {f:dict(mean=float(np.mean([r[f] for r in rr])),
                                     median=float(np.median([r[f] for r in rr])),
                                     p90=float(np.percentile([r[f] for r in rr],90)),
                                     maximum=float(max(r[f] for r in rr))) for f in fields}
            offline['%s_%d_%d'%(arm,stage,task)] = z

gradients = {}
for arm in 'CRG':
    for stage in range(7,10):
        rr = [r for r in light['GRADIENTS_'+arm+'.json'] if r['stage']==stage]
        assert len(rr)==4
        terms = ['bc','ifa_weighted','exec_weighted','route_weighted']+(['anchor_weighted'] if arm=='G' else [])
        gradients['%s_%d'%(arm,stage)] = {
            term: dict(median_l2=float(np.median([r[term]['l2'] for r in rr])),
                       max_l2=max(r[term]['l2'] for r in rr),
                       median_ratio_to_bc=float(np.median([r[term]['l2']/max(r['bc']['l2'],1e-300) for r in rr])),
                       nonzero_points=sum(r[term]['l2']>1e-12 for r in rr)) for term in terms}
teacher = {key:dict(tau=value['tau'],count=value['count'],source_sha256=value['source_sha256'],
                   seconds=value.get('seconds'),forward_batches=value['forward_batches'])
           for key,value in light.items() if key.startswith('TEACHER_')}
out = dict(status='ACCEPTED_LOCAL_SCALAR_ANALYSIS',
           generated=datetime.datetime.now().astimezone().isoformat(),
           acceptance_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),
           common_windows=common_windows,
           offline_by_stage_and_task=offline, gradient_measurement_summary=gradients,
           teacher_preparation=teacher,
           limits=['Replay populations and rolling teachers depend on each trained arm; no closed-loop causal localization.',
                   'Four gradient measurement points per stage; norms do not measure gradient alignment or conflict.',
                   'No additional predictions, training updates or environment episodes were run by this script.'])
save(D/'MECHANISM_DIAGNOSTICS.json',out)
lines=['# 完整闭环矩阵与配对变化','',
       '|训练阶段|评估任务|C|R|G|G−C|G−R|R−C|','|---|---|---:|---:|---:|---:|---:|---:|']
for stage in range(6,10):
    for task in range(6,stage+1):
        key='%d_%d'%(stage,task)
        vals=[a['matrix'][arm][key]['percent'] for arm in 'CRG']
        delta=[a['paired'][pair][key]['delta_pp'] for pair in ['G-C','G-R','R-C']]
        lines.append('|%d|%d|%g%%|%g%%|%g%%|%+g|%+g|%+g|'%(stage,task,*vals,*delta))
lines += ['','SF为参考成功而候选失败；FS为参考失败而候选成功。同初态和逐轨迹噪声配对，不表示两策略的实际观测历史相同。','',
          '|比较|阶段/任务|SF|FS|净变化(pp)|','|---|---|---:|---:|---:|']
for pair,cells in a['paired'].items():
    for key,cell in cells.items():
        lines.append('|%s|%s|%d|%d|%+g|'%(pair,key,len(cell['reference_success_to_candidate_failure']),len(cell['reference_failure_to_candidate_success']),cell['delta_pp']))
lines += ['','## 最终阶段旧任务回放诊断','',
          '|组别|任务|合法窗口|执行专家误差均值|P90|regret均值|非最优分量数|教师距离均值|方向锚均值|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for task in range(6,9):
    for arm in 'CRG':
        z=offline['%s_9_%d'%(arm,task)]; s=z['statistics']
        lines.append('|%s|%d|%d|%.8g|%.8g|%.8g|%d|%.8g|%.8g|'%(arm,task,z['count'],s['expert_exec']['mean'],s['expert_exec']['p90'],s['regret']['mean'],z['nonbest_count'],s['teacher_distance_exec']['mean'],s['directional_exec']['mean']))
(D/'RESULT_TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],offline_groups=len(offline),gradient_groups=len(gradients)),indent=2))
