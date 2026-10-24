import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'20261024_align_behavior_drift'
RA=next(R.glob('ra_drift_readonly_*/SUMMARY.json')).parent;AL=next(P.glob('drift_readonly_v2_*/SUMMARY.json')).parent
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];sources={}
for seed in [41,42,43]:
 a=js(AL/('seed_%d.json'%seed));r=js(RA/('seed_%d.json'%seed))
 assert [(x['sample'],x['sample_sha256'],x.get('next_action_sha256'),x['bc_valid']) for x in a['rows']]==[(x['sample'],x['sample_sha256'],x.get('next_action_sha256'),x['bc_valid']) for x in r['rows']]
 for method,root,d in [('RA',RA,r),('ALIGN',AL,a)]:
  s=js(root/'SUMMARY.json')[str(seed)];valid=[x for x in d['rows'] if x['bc_valid']];flips=[x for x in valid if x['gripper_strict_sign_flip']]
  rows.append({'seed':seed,'method':method,'switches':s['all']['mode_switch_count'],'flip_all':s['all']['sign_flip_count'],'flip_valid':len(flips),'flip_expert_huber_improved':sum(x['expert']['grouped_huber']['delta']<0 for x in flips),'flip_expert_huber_worsened':sum(x['expert']['grouped_huber']['delta']>0 for x in flips),'mean_huber_delta':s['expert']['grouped_huber']['delta']['mean'],'mean_mixture_nll_delta':s['expert']['mixture_nll']['delta']['mean'],'closed_loop':s['closed_loop_existing_percent']})
  for name in ['seed_%d.json'%seed,'SUMMARY.json','LOCAL_AUDIT.json']:sources[str(root/name)]=sha(root/name)
out={'status':'PASS_COMMON_WINDOW_OFFLINE_COMPARISON','common_windows':1356,'common_valid':1283,'rows':rows,'sources':sources,'not_causal':True}
(R/'OFFLINE_COMPARISON.json').write_text(json.dumps(out,indent=2),encoding='utf8')
report=['# RA 与 ALIGN：同窗口漂移诊断','', '三seed共1356个同字节窗口，1283个共同合法下一步标签；RA新增6CP/174批/2712模型窗口，0更新/环境。ALIGN复用已接受结果。','', '|seed|方法|分量切换/452|夹爪翻转(有效)|有效翻转误差改善/恶化|平均Huber变化|混合NLL变化|既有Task8闭环Stage8→9|','|---|---|---|---|---|---|---|---|']
for x in rows:
 c=x['closed_loop'];report.append('|%s|%s|%s|%s (%s)|%s/%s|%.8f|%+.5f|%g→%g%%|'%(x['seed'],x['method'],x['switches'],x['flip_all'],x['flip_valid'],x['flip_expert_huber_improved'],x['flip_expert_huber_worsened'],x['mean_huber_delta'],x['mean_mixture_nll_delta'],c['stage8_task8'],c['stage9_task8']))
report+=['','RA同样存在分量切换和夹爪纠错，seed41/42平均下一步Huber也改善；seed43平均Huber恶化，三seed混合下一步NLL均恶化。不能笼统说两方法所有离线指标都改善。RA闭环在seed41/43反而提升，seed42下降。这些现象不构成ALIGN特有退化机制，也不能把所有夹爪翻转视为应抑制的错误。','', 'RA原监督a[u]、ALIGN监督a[u+1]；此处为共同a[u+1]标签上的诊断，不将RA巨大所选Normal NLL当实现错误或训练损失。原近零方差保留，未裁剪。窗口重叠、三seed且同数据反复探索，不作窗口独立显著性或因果宣称。','', '新配对轨迹诊断尚待准入与运行，当前总状态PARTIAL；保护训练未启动。']
(R/'OFFLINE_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf8')
print(json.dumps(rows,indent=2))
