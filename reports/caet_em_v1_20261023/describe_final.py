"""Report complete arms only, including negative cells and unequal auxiliary cost."""
import sys,json
from pathlib import Path
F=Path(sys.argv[1]);S=Path(sys.argv[2]);data=json.loads((F/'LOCAL_FINAL_AUDIT.json').read_text());assert data['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE';meta=S/'metadata';cost={};diagnostics={}
def js(p):return json.loads(p.read_text())
codec=js(meta/'shared/CODEC_TRAINING_AND_VALIDATION.json');cost['codec']=dict(epochs=len(codec['epochs']),updates=codec['optimizer_steps'],seconds=sum(r['seconds'] for r in codec['epochs']),selected_epoch=codec['selected_epoch'])
for arm in ('B0','B1','B2'):
 rows=[r for t in range(6,10) for r in js(meta/arm/'epochs'/('task%d.json'%t))];cost[arm]=dict(main_epochs=len(rows),main_updates=sum(r['batches'] for r in rows),main_epoch_seconds=sum(r['seconds'] for r in rows),action_valid=sum(r['action_valid'] for r in rows),action_masked=sum(r['action_invalid'] for r in rows),policy_windows=sum(r['policy_windows'] for r in rows),retention_decoder_windows=sum(r['extra_retention_decoder_windows'] for r in rows),transfer_anchors=sum(r['transfer_anchors'] for r in rows),retained_steps=sum(r['retained_steps'] for r in rows))
 stages={}
 for t in range(6,10):
  raw=js(meta/arm/'diagnostics'/('task%d.json'%t));groups={}
  for row in raw['rows']:
   group='current' if row['current'] else 'old_RA';g=groups.setdefault(group,{})
   for k,v in row['metrics'].items():
    r=g.setdefault(k,dict(sum=0.,count=0));r['sum']+=v['sum'];r['count']+=v['count']
  stages[str(t)]={g:{k:r['sum']/r['count'] if r['count'] else None for k,r in v.items()} for g,v in groups.items()}
 diagnostics[arm]=stages
negative={}
for a,b in [('B1','B0'),('B2','B1'),('B2','B0')]:negative[a+'-'+b]={k:5*(v-data['arms'][b]['success_counts'][k]) for k,v in data['arms'][a]['success_counts'].items() if v<data['arms'][b]['success_counts'][k]}
(F/'COST_AUDIT.json').write_text(json.dumps(dict(cost=cost,not_equal_compute=True,extra_storage_not_equal_memory=True,diagnostics_and_admission_separate=True),indent=2));(F/'OFFLINE_DIAGNOSTICS.json').write_text(json.dumps(diagnostics,indent=2));(F/'NEGATIVE_CELLS.json').write_text(json.dumps(negative,indent=2))
lines=['# CAET-EM-v1 OBJECT seed41 完整结果','','三组完整本地独立验收；单seed探索，不宣称显著性、防遗忘保证或跨套件泛化。旧位置探针A失败保留，此路线不放宽或改写该门。','','|组|AUC %|FWT %|NBT %（低好）|OldFinal %|FinalAll %|','|---|---:|---:|---:|---:|---:|']
for a in ('B0','B1','B2'):lines.append('|'+a+'|'+'|'.join('%.6f'%data['arms'][a]['metrics'][k]['percent'] for k in ('AUC','FWT','NBT','OldFinal','FinalAll'))+'|')
lines+=['','B1−B0检验效果接口；B2−B1检验条件迁移和功能保护组合。所有负单元见NEGATIVE_CELLS.json，完整10cell见LOCAL_FINAL_AUDIT.json，不能只报AUC或只选最好的组。','','旧ALIGN41 AUC81.458333%、OldFinal70%、FinalAll75%仅作外部参考；共享读出和四步监督已改变，不作严格单因素比较。','','主更新三组316500、正式episode600；额外codec训练、教师及诊断前向见COST_AUDIT.json与原阶段证据，不称等算力/等总内存。任何新增seed或消融须另行决定，无自动扩展。']
(F/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(F/'FINAL_REPORT.md')
