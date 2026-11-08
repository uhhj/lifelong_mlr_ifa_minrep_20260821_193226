"""Count existing acquisition outcomes and cache contributors; no new experiments."""
from pathlib import Path
from collections import Counter
import json,csv,hashlib,statistics,datetime
R=Path(__file__).resolve().parent;W=R.parents[1];E=W/'execution'
SOURCES={};LIGHT={};rows=[];stages=[];episodes=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):SOURCES[p.relative_to(W).as_posix()]=sha(p);return json.loads(p.read_text(encoding='utf8'))
def csvout(name,data):
    with (R/name).open('w',encoding='utf8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(data[0]));wr.writeheader();wr.writerows(data)
def emit(name,obj):(R/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
specs=[
('S','OBJECT',.1,'20261028_support_gate_full_s_object/final_readonly_20261029_201235'),
('S','GOAL',.1,'20261029_support_gate_full_s_goal/final_readonly_20261030_111546'),
('Q',None,.1,'20261031_full_q_vs_latest_s_two_suites/final_readonly_20261101_032229'),
('S','OBJECT',None,'20261101_lambda_r_object_sensitivity/final_acceptance_20261103_v1/final_readonly_20261103_124306'),
('S','GOAL',None,'20261103_lambda_r_goal_sensitivity/final_readonly_20261104_122613')]
for arm,fixedsuite,fixedlambda,path in specs:
    root=E/path;e=read(root/'EVIDENCE.json');accept=read(root/'LOCAL_FINAL_AUDIT.json')
    assert accept['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE',accept['status']
    cases={f'seed{s}':dict(suite=fixedsuite,seed=s,lambda_R=.1,key=str(s)) for s in [41,42,43]} if 'seeds' in e else {}
    if not cases:
        for name in e['cases']:
            prefix,seed=name.split('_');suite=fixedsuite or prefix
            cases[name]=dict(suite=suite,seed=int(seed),lambda_R=fixedlambda if fixedlambda is not None else {'L005':.05,'L020':.2}[prefix],key=name)
    def item(case,name):
        key=case+'/'+name;record=e['light'][key];txt=record['text'];h=hashlib.sha256(txt.encode('utf8')).hexdigest()
        assert h==record['sha256'] and len(txt.encode('utf8'))==record['bytes'],key
        LIGHT[path+'/'+key]=h
        return json.loads(txt)
    for name,meta in cases.items():
        base=dict(method=arm,suite=meta['suite'],seed=meta['seed'],lambda_R=meta['lambda_R']);memories={};runrows={}
        for t in range(6,10):
            outcomes=item(name,f'COLLECTION_OUTCOMES_{arm}_{t}.json');memory=item(name,f'MEMORY_{arm}_{t}.json')
            assert len(outcomes)==10 and {o['initial_index'] for o in outcomes}==set(range(20,30))
            assert all(o['task']==t and o['arm']==arm and o['attempt']==1 and o['purpose']=='training_collection_not_formal_evaluation' for o in outcomes)
            successful={o['initial_index']-20 for o in outcomes if o['success']}
            contributors={r['trajectory'] for r in memory['rows']}
            assert len(memory['rows'])==memory['count'] and contributors<=successful
            assert all(r['initial_index']==r['trajectory']+20 and r['task']==t for r in memory['rows'])
            q=len(contributors)/10
            row=dict(**base,task=t,attempts=10,successful_trajectories=len(successful),cache_contributing_trajectories=len(contributors),q=q,cache_windows=memory['count'],used_in_later_training=t<9,later_stage_count=9-t,successful_ids=','.join(map(str,sorted(successful))),contributor_ids=','.join(map(str,sorted(contributors))),source_snapshot=path)
            rows.append(row);runrows[t]=row;memories[t]=contributors
            for o in outcomes:episodes.append(dict(**base,task=t,initial_index=o['initial_index'],success=o['success'],contributes_cache=o['initial_index']-20 in contributors,attempt=o['attempt']))
        cpu=e.get('seeds',e.get('cases'))[meta['key']]
        for stage in [7,8,9]:
            inp=item(name,f'TRAIN_INPUT_{arm}_{stage}.json')['memory'];stagebinding=item(name,f'STAGE_{arm}_{stage}.json')['memory_bindings']
            assert set(inp)==set(stagebinding)==set(map(str,range(6,stage)))
            for t in range(6,stage):
                for bound in [inp[str(t)],stagebinding[str(t)]]:
                    assert bound['support']==runrows[t]['q'] and set(bound['support_trajectories'])==memories[t] and bound['count']==runrows[t]['cache_windows']
            qs=[runrows[t]['q'] for t in range(6,stage)]
            gate_blocks=gate_accepts=None
            if arm=='S':
                g=cpu['gate_audit']['by_stage'][str(stage)]
                assert g['support']=={str(t):runrows[t]['q'] for t in range(6,stage)}
                gate_blocks=g['blocks'];gate_accepts=g['Q_accepts']
            active=[q for q in qs if q>0]
            stages.append(dict(**base,stage=stage,q6=qs[0],q7=qs[1] if len(qs)>1 else '',q8=qs[2] if len(qs)>2 else '',nonempty_old_tasks=len(active),any_q_below_one=any(q<1 for q in active),all_active_q_one=all(q==1 for q in active),unequal_active_q=len(set(active))>1,uniform_nonempty_mean=statistics.mean(active) if active else 0,minimum_q=min(active) if active else 0,maximum_q=max(active) if active else 0,gate_blocks=gate_blocks,Q_accepts=gate_accepts))
assert len(rows)==96 and len(episodes)==960 and len(stages)==72
assert len({(r['method'],r['suite'],r['lambda_R'],r['seed'],r['task']) for r in rows})==96
csvout('task_success_counts.csv',rows);csvout('collection_episode_outcomes.csv',episodes);csvout('training_stage_support.csv',stages)

def aggregate(rr,ss):
    active=[r for r in rr if r['used_in_later_training']];totalpairs=sum(r['later_stage_count'] for r in active)
    g=[s for s in ss if s['gate_blocks'] is not None]
    return dict(task_collections=len(rr),attempts=sum(r['attempts'] for r in rr),successes=sum(r['successful_trajectories'] for r in rr),contributors=sum(r['cache_contributing_trajectories'] for r in rr),success_cache_count_mismatches=sum(r['successful_trajectories']!=r['cache_contributing_trajectories'] for r in rr),q_one_all_tasks=sum(r['q']==1 for r in rr),used_task_collections=len(active),q_one_used=sum(r['q']==1 for r in active),q_below_one_used=sum(r['q']<1 for r in active),q_below_one_used_fraction=sum(r['q']<1 for r in active)/len(active),q_histogram_used=dict(sorted(Counter(str(r['q']) for r in active).items())),mean_q_used=statistics.mean(r['q'] for r in active),training_stages=len(ss),stages_with_any_q_below_one=sum(s['any_q_below_one'] for s in ss),stages_with_unequal_active_q=sum(s['unequal_active_q'] for s in ss),multi_old_task_stages=sum(s['nonempty_old_tasks']>=2 for s in ss),old_task_stage_pairs=totalpairs,q_below_one_old_task_stage_pairs=sum(r['later_stage_count'] for r in active if r['q']<1),old_task_stage_weighted_mean_q=sum(r['q']*r['later_stage_count'] for r in active)/totalpairs,gate_blocks=sum(s['gate_blocks'] for s in g) if g else None,gate_blocks_in_q_below_one_stages=sum(s['gate_blocks'] for s in g if s['any_q_below_one']) if g else None)
groups={}
for method,lam in [('S',.1),('Q',.1),('S',.05),('S',.2)]:
    for suite in ['OBJECT','GOAL']:
        match=lambda x:x['method']==method and x['lambda_R']==lam and x['suite']==suite
        groups[f'{method}_{lam}_{suite}']=aggregate([r for r in rows if match(r)],[s for s in stages if match(s)])
primary=[r for r in rows if r['method']=='S' and r['lambda_R']==.1]
primary_stages=[s for s in stages if s['method']=='S' and s['lambda_R']==.1]
summary=dict(primary_latest_S=aggregate(primary,primary_stages),groups=groups,scope='Accepted complete S lambda0.1, Q lambda0.1, S lambda0.05/0.2; no historical canceled forks or duplicated progress snapshots; Task9 separated from training exposure.',all_success_counts_equal_cache_contributor_counts=all(r['successful_trajectories']==r['cache_contributing_trajectories'] for r in rows))
emit('SUMMARY.json',summary)

lines=['# 成功采集与支持度实际使用统计','',
'只读统计已接受的完整实验：最新S（lambda=.1）、完整Q（lambda=.1）、S的lambda=.05/.2；两个套件各三个种子。共24个run、96个任务采集组、960次既有训练采集。未混入正式评估初态0–19、旧取消机制分叉、进度快照的重复记录或工程诊断。','',
'## 最新S：主决策依据','',
'各单元为10次训练采集的成功轨迹数；本次96组中成功轨迹数与实际贡献缓存的独立轨迹数全部相同，因此q=表中数值/10。Task9仍完整披露，但不用于后续训练，不能算作校准暴露。','',
'|套件|seed|Task6|Task7|Task8|Task9（不再训练）|', '|---|---:|---:|---:|---:|---:|']
for suite in ['OBJECT','GOAL']:
    for seed in [41,42,43]:
        rr=sorted([r for r in primary if r['suite']==suite and r['seed']==seed],key=lambda r:r['task'])
        lines.append('|'+suite+'|'+str(seed)+'|'+'|'.join(str(r['successful_trajectories']) for r in rr)+'|')
lines += ['', '## 只计真正用于后续训练的Task6–8','', '|配置|套件|q=1 / 任务采集组|平均q|存在q<1的训练阶段|多旧任务且q不同的阶段|','|---|---|---:|---:|---:|---:|']
for key,g in groups.items():
    method,lam,suite=key.split('_')
    lines.append(f'|{method}, lambda={lam}|{suite}|{g["q_one_used"]}/{g["used_task_collections"]}|{g["mean_q_used"]:.4f}|{g["stages_with_any_q_below_one"]}/{g["training_stages"]}|{g["stages_with_unequal_active_q"]}/{g["multi_old_task_stages"]}|')
lines += ['', '“存在q<1”表示该阶段的实际保持系数至少一项不同于q=1，不代表已证明产生闭环收益。阶段不等长，也不是独立实验重复。“q不同”仅指该阶段至少两个非空旧任务采用不同权重；单旧任务阶段只有整体缩放，没有任务间分配。','',
'## 对新增消融建议的修正','',
'1. 用户的近饱和担心在最新S的GOAL上基本成立：9个被后续使用的任务采集组中7个q=1，另外为0.9和0.8；等权平均q约0.967。seed41在三个保持训练阶段的全部权重都是1，其已观察训练路径上支持度校准没有改变系数。Task9三seed均8/10不应被误计为训练中校准起作用。GOAL不宜默认作为新增校准消融的优先对象。',
'2. OBJECT不能视为近似全1：Task6三个seed分别q=.2/.2/.6，并连续进入Task7、8、9的保持目标；其余主要接近1。最新S的9个保持训练阶段均存在q<1，六个多旧任务阶段均有不同q。按任务采集组计，平均q约0.767；按旧任务—训练阶段使用次数计约0.656，二者都不是实际梯度加权均值。',
'3. 因而修正上一轮默认建议两套件共6run：当前先不启动新实验；若确需验证逐任务校准的独立贡献，优先考虑OBJECT三个seed的一个预先固定变体，并完整报告三seed。选择依据是机制暴露与权重差异，而非按最终成绩挑任务。GOAL可先报告近饱和事实，不为凑齐表格盲目追加。',
'4. 若使用统一平均q对照，单旧任务阶段与原q完全一样，差异从Task8开始；q=1对照则从Task7开始改变保持强度。两者回答的问题不同。该统计不会替代任何闭环反事实，也不能预言消融一定有显著收益。',
'5. Q和不同lambda的完整数据说明q的饱和程度随训练路径变化。它们是补充诊断，不把不同run的缓存当成同一份、不把其他配置中的q分布直接套给最新S。q是十次训练采集的经验支持量，不是经过概率校准的逐动作可靠性，也不据此给收益概率。','',
'## 其他已完成配置的完整计数','']
for method,lam in [('Q',.1),('S',.05),('S',.2)]:
    lines += [f'### {method}, lambda={lam}','','|套件|seed|Task6|Task7|Task8|Task9|','|---|---:|---:|---:|---:|---:|']
    for suite in ['OBJECT','GOAL']:
        for seed in [41,42,43]:
            rr=sorted([r for r in rows if (r['method'],r['lambda_R'],r['suite'],r['seed'])==(method,lam,suite,seed)],key=lambda r:r['task'])
            lines.append('|'+suite+'|'+str(seed)+'|'+'|'.join(str(r['successful_trajectories']) for r in rr)+'|')
    lines.append('')
lines += ['## 核验与产物','',
'- 从最终已接受快照逐条读取COLLECTION_OUTCOMES，核每组10条、initial_index20–29、attempt1、training_collection目的；共960条。',
'- 从MEMORY行去重trajectory，核其initial_index对应、仅来自成功采集、缓存窗口数一致。不能把128窗口当128条成功轨迹。',
'- 逐阶段核TRAIN_INPUT与STAGE的support、support_trajectories、count；S再与最终独立门控审计的support逐项核对。',
'- 对内嵌轻量记录校验原文本SHA与字节数；对五份最终证据及LOCAL_FINAL_AUDIT绑定文件SHA，统计前后再核不变。',
'- task_success_counts.csv保存96组；collection_episode_outcomes.csv保存960条允许发布的成功/贡献记录；training_stage_support.csv保存72阶段实际权重向量。SUMMARY.json明确各分母与分组。',
'- 本次训练、模型前向、优化器、环境调用均为0；没有恢复旧机制或部署新消融。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
assert all(sha(W/p)==h for p,h in SOURCES.items())
emit('REVIEW_AUDIT.json',dict(status='PASS_SUCCESS_SUPPORT_INVENTORY',observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),sources=SOURCES,embedded_light_sha256=LIGHT,report_sha256=sha(R/'REPORT.md'),artifacts={p.name:sha(p) for p in R.glob('*.csv')},runs=24,task_collections=96,episodes=960,stages=72,success_cache_mismatches=0,new_training_runs=0,new_environment_episodes=0,new_optimizer_calls=0,source_inputs_unchanged=True))
print(json.dumps(summary,indent=2))
