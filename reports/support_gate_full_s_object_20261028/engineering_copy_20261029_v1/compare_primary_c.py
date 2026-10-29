"""Primary new-S versus accepted full-sequence C comparison; existing records only."""
from pathlib import Path
import hashlib,json,statistics
N=Path(__file__).resolve().parent.parent;D=N/'final_readonly_20261029_201235'
C=N.parent/'20261027_c_goal42_43_object41_42_43/final_readonly_20261027_222119'
js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sa=js(D/'LOCAL_FINAL_AUDIT.json');sv=js(D/'EVIDENCE.json');ca=js(C/'LOCAL_FINAL_AUDIT.json');cv=js(C/'EVIDENCE.json')
assert sa['status']==ca['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
assert sa['evidence_sha256']==sha(D/'EVIDENCE.json') and ca['source_sha256']==sha(C/'EVIDENCE.json')
keys=['AUC','FWT','NBT','OldFinal','FinalAll'];results={};paired=[];negative=[]
for seed in ['41','42','43']:
    case='OBJECT_'+seed
    sr=json.loads(sv['light']['seed'+seed+'/OUTCOMES.json']['text']);cr=json.loads(cv['light'][case+'/OUTCOMES.json']['text'])
    index=lambda rows:{(r['stage'],r['task'],r['initial_index']):r for r in rows}
    ss,cc=index(sr),index(cr);assert len(ss)==len(cc)==200 and set(ss)==set(cc)
    for k,x in ss.items():
        y=cc[k];assert x['attempt']==y['attempt']==1
        assert all(x[f]==y[f] for f in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed'])
        paired.append(dict(seed=seed,stage=k[0],task=k[1],initial_index=k[2],C_success=y['success'],S_success=x['success'],initial_sha256=x['initial_sha256'],noise_sha256=x['noise_sha256'],env_seed=x['env_seed']))
    cells={}
    for stage in range(6,10):
        for task in range(6,stage+1):
            x=[r for r in paired if r['seed']==seed and r['stage']==stage and r['task']==task]
            c=sum(r['C_success'] for r in x);s=sum(r['S_success'] for r in x);key=f'{stage}_{task}'
            assert c==ca['matrix'][case][key]['successes'] and s==sa['seeds'][seed]['matrix'][key]['successes']
            cells[key]=dict(C_percent=c*5,S_percent=s*5,delta_pp=(s-c)*5,C_success_S_fail=sum(r['C_success'] and not r['S_success'] for r in x),C_fail_S_success=sum(not r['C_success'] and r['S_success'] for r in x))
            if s<c:negative.append(dict(seed=seed,cell=key,**cells[key]))
    cm=ca['metrics'][case];sm=sa['seeds'][seed]['metrics']
    results[seed]=dict(C_metrics=cm,S_metrics=sm,delta_metrics_pp={k:sm[k]['percent']-cm[k]['percent'] for k in keys},cells=cells)
means={method:{k:statistics.mean(r[method+'_metrics'][k]['percent'] for r in results.values()) for k in keys} for method in ['C','S']}
delta={k:means['S'][k]-means['C'][k] for k in keys}
out=dict(status='PASS_ACCEPTED_C_S_600_EPISODE_PAIRING',primary_reference='Completed full-sequence C OBJECT41/42/43, not the canceled single-stage mechanism fork and not internal 10-step C candidates.',C_final_audit_sha256=sha(C/'LOCAL_FINAL_AUDIT.json'),C_evidence_sha256=sha(C/'EVIDENCE.json'),S_final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),S_evidence_sha256=sha(D/'EVIDENCE.json'),paired_episodes=600,paired_fields=['initial_file_sha256','initial_sha256','noise_sha256','env_seed'],seeds=results,means=means,mean_delta_pp=delta,negative_cells=negative,paired_outcomes=paired,costs=dict(C_production_optimizer=316500,S_production_optimizer=585900,C_training_collection=0,S_training_collection=120,formal_episodes_per_method=600),limitations=['Matched evaluation seeds/initial bytes/noise do not make the visited closed-loop observations identical.', 'S has additional collection, retention and dual candidates; this is not an equal-interaction/equal-compute attribution of the gate alone.', 'Three development seeds and 20 episodes per cell do not establish statistical significance.', 'No training, model forward, collection or evaluation was run for this comparison.'])
(D/'PRIMARY_C_COMPARISON.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(means=means,delta=delta,seeds={s:r['delta_metrics_pp'] for s,r in results.items()},negative=negative),ensure_ascii=False,indent=2))
lines=['## 主对照更正：最新完整S对完整C','','用户指出后重新核对：主对照应为已于10月27日独立验收的完整C OBJECT41/42/43。上一版遗漏了这个已存在的主对照，把历史原S作为主要比较对象；这是报告选择错误，不是缺少C结果。原S比较保留为版本参考，不能替代C。这里的C保留原专家执行成本与margin路由，关闭S保持项；它不同于已取消的单阶段机制分叉，也不同于新S内部每块C候选。','','已对全部600回合逐项核验相同seed、阶段/任务/初态编号、初态文件与实际字节SHA、动作噪声SHA和环境seed，均一致，各唯一attempt=1。两套已接受原证据SHA和全部配对正负单元见PRIMARY_C_COMPARISON.json。仅复算已有记录，未重训、重评、补采或重新开启监控。','','|指标（三seed均值）|C|最新完整S|S−C（百分点）|','|---|---:|---:|---:|']
for k in keys:lines.append('|%s|%.4f|%.4f|%+.4f|'%(k,means['C'][k],means['S'][k],delta[k]))
lines+=['','相对C，最新S平均AUC提高1.4931个百分点，旧任务最终均值提高3.8889个百分点，最终全任务均值提高2.50个百分点；获得阶段均值FWT下降0.8333个百分点。最终新Task9均值从C的98.3333%降至S的96.6667%（−1.6667pp）。总体表现为部分旧任务收益伴随部分学习/任务代价，不是全面胜出。','','seed41最终Task6为C50%→S70%，Task8为100%→95%；seed42最终Task6为35%→65%，Task8及Task9均100%→90%；seed43最终Task6两者均30%，该关键遗忘问题相对C没有改善。seed43最终Task8由85%→90%、Task9由95%→100%，使FinalAll由78.75%→80%。三个seed的FinalAll分别增加3.75、2.50、1.25pp，但AUC在seed41/42分别下降0.4167/2.3958pp，seed43上升7.2917pp，不能用最终均值遮住整个学习过程的负结果。','','|seed|C FinalAll|S FinalAll|ΔFinalAll|ΔAUC|','|---|---:|---:|---:|---:|']
for seed,r in results.items():lines.append('|%s|%.2f|%.2f|%+.2f|%+.4f|'%(seed,r['C_metrics']['FinalAll']['percent'],r['S_metrics']['FinalAll']['percent'],r['delta_metrics_pp']['FinalAll'],r['delta_metrics_pp']['AUC']))
lines+=['','全部相对C的负单元：','','|seed|阶段/任务|C|S|Δpp|C成功/S失败|C失败/S成功|','|---|---|---:|---:|---:|---:|---:|']
for r in negative:lines.append('|%s|%s|%.1f|%.1f|%+.1f|%d|%d|'%(r['seed'],r['cell'],r['C_percent'],r['S_percent'],r['delta_pp'],r['C_success_S_fail'],r['C_fail_S_success']))
lines+=['','比较边界：C与S评估配对已核实，但S增加120次训练采集，生产optimizer从C的316500增至585900，另有各自准入/工程成本。因此可以报告对C的闭环收益及代价，不能把差异单独归因于支持度或门控，也不能称等交互、等算力或统计显著。下文原S比较只用于版本演进参考；“旧任务均值下降”仅指相对原S，不适用于相对C。','']
section='\n'.join(lines)
(D/'PRIMARY_C_COMPARISON.md').write_text(section,encoding='utf8')
for name in ['FINAL_REPORT.md','REPORT.md']:
    p=N/name;text=p.read_text(encoding='utf8');head,body=text.split('\n',1)
    text=head+'\n\n'+section+'\n'+body.lstrip('\n')
    text=text.replace('因此本轮没有证明旧任务保持整体改善，更不能仅凭FinalAll上升就宣称支持度门控全面有效。','这一句仅比较历史原S：相对原S的旧任务最终均值未提高；相对主对照C则提高3.8889个百分点，仍不能仅凭FinalAll上升宣称全面有效。')
    text=text.replace('整体旧任务最终均值没有改善。','相对历史原S的旧任务最终均值没有改善；相对主对照C提高3.8889个百分点。')
    p.write_text(text,encoding='utf8')
