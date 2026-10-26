"""Summarize accepted S41/42/43 and historical baselines without new model calls."""
import json,sys,hashlib,statistics
from pathlib import Path
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_FINAL_AUDIT.json');assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
ref=js(R/'REFERENCE_RESULTS.json');S={str(seed):a['seeds'][str(seed)] for seed in (42,43)};S['41']=ref['S41']
base=ref['baselines']['runs'];differences={};negatives=[]
for seed in (41,42,43):
 ss=S[str(seed)];differences[str(seed)]={}
 for rr in [x for x in base if x['seed']==seed]:
  dm={k:ss['metrics'][k]['percent']-x['percent'] for k,x in rr['metrics'].items()}
  cells={k:ss['matrix'][k]['percent']-x for k,x in rr['matrix'].items()}
  differences[str(seed)][rr['method']]=dict(metric_delta_pp=dm,cell_delta_pp=cells,causal_paired_comparison=False)
  negatives.extend(dict(seed=seed,reference=rr['method'],cell=k,delta_pp=x) for k,x in cells.items() if x<0)
stats={k:dict(mean=statistics.mean(S[str(seed)]['metrics'][k]['percent'] for seed in (41,42,43)),sample_sd=statistics.stdev(S[str(seed)]['metrics'][k]['percent'] for seed in (41,42,43))) for k in ['AUC','FWT','NBT','OldFinal','FinalAll']}
out=dict(status='ACCEPTED_FINITE_EXTENSION_NO_AUTOMATIC_EXPANSION',S=S,three_seed_statistics=stats,historical_comparisons=differences,all_negative_cells=negatives,seed41_original_support=ref['seed41_original_support'],final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),reference_sha256=sha(R/'REFERENCE_RESULTS.json'),limits=a['limits'])
(D/'COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
lines=['# S：GOAL seed41/42/43有限扩展结果','','两新seed完成独立验收；科学正负结果均保留。未自动扩展种子或调参。','','|seed|AUC|FWT|NBT（低好）|OldFinal|FinalAll|','|---|---:|---:|---:|---:|---:|']
for seed in (41,42,43):lines.append('|'+str(seed)+'|'+'|'.join('%.4f'%S[str(seed)]['metrics'][k]['percent'] for k in stats)+'|')
lines+=['|均值±样本标准差|'+'|'.join('%.4f±%.4f'%(x['mean'],x['sample_sd']) for x in stats.values())+'|','','完整矩阵、所有负单元和逐seed历史基线差值见COMPARISON.json；离线逐窗口重算见LOCAL_FINAL_AUDIT及RECOMPUTED文件。不同seed不能直接按初态索引解释配对SF/FS，历史基线也没有同一动作随机流及相同额外交互。缓存来自成功轨迹不等于每一步最优；低离线误差不构成闭环因果解释。','',*a['limits']]
(D/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(dict(status=out['status'],three_seed_statistics=stats),indent=2))
