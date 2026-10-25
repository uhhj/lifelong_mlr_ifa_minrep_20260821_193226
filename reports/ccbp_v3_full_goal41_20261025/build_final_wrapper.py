"""New external final path. AST only until all fresh C/R/G execution finishes."""
from pathlib import Path
import ast
R=Path(__file__).resolve().parent;P=R.parent/'20261024_ccbp_v2_full'
oldroot='/data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1_engineering02'
newroot='/data/mlr_ifa/experiment_workspaces/CCBP_V3_FULL_GOAL41_20261025_v1'
s=(P/'final_readonly.py').read_text(encoding='utf8').replace(oldroot,newroot)
start=s.index("if (Q/'REPAIR_CONTINUATION.json').exists():");end=s.index('manifests={}',start);s=s[:start]+s[end:]
s=s.replace("task6['B']","task6['C']").replace('==800','==600').replace('episodes=800','episodes=600')
s=s.replace('strict={};diags={}', 'from directional_prototype import directional_anchor\nstrict={};diags={}')
needle='   for j,i in enumerate(chunk):'
insert="""   directional,fraction=directional_anchor(smus,sstd,tmus[ii,tk],tstd[ii,tk],y,scale)
   # Independent identity from three squared distances; export scalars only.
   ctotal=sg.sum(-1);dtotal=anchor.sum(-1);tctotal=tg.sum(-1)[ii,tk]
   dot=(dtotal+tctotal[:,None]-ctotal)/2
   alpha=(dot/tctotal[:,None].clamp_min(1e-300)).clamp(0,1)
   gram=dtotal-2*alpha*dot+alpha.square()*tctotal[:,None]
   assert torch.allclose(directional,gram,atol=1e-9,rtol=1e-9)
   assert torch.allclose(fraction,alpha,atol=1e-9,rtol=1e-9)
"""
s=s.replace(needle,insert+needle)
s=s.replace("quality_weight=w['weight'],frozen_teacher_cost=w['teacher_cost']", "quality_weight=w['weight'],frozen_teacher_cost=w['teacher_cost'],directional_distances=directional[j].tolist(),projection_fractions=fraction[j].tolist()")
# Parent final acceptance is bound as a local lightweight document in new contract.
s=s.replace("contract=js(Q/'CONTRACT.json');assert", "parent_acceptance=js(Q/'PARENT_LOCAL_ACCEPTANCE.json')\nassert parent_acceptance['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'\ncontract=js(Q/'CONTRACT.json');assert")
ast.parse(s);(R/'final_readonly.py').write_text(s,encoding='utf8')
s=(P/'audit_final_local.py').read_text(encoding='utf8')
s=s.replace("'BCNV'","'CRG'").replace('==70','==60').replace('==800','==600')
s=s.replace("==48", "==(60 if arm=='G' else 48)")
s=s.replace("for k in ['bc','ifa_weighted','exec_weighted','route_weighted']", "for k in (['bc','ifa_weighted','exec_weighted','route_weighted']+(['anchor_weighted'] if arm=='G' else []))")
s=s.replace("[('V','C'),('V','B'),('V','N'),('N','C'),('B','C')]", "[('G','C'),('G','R'),('R','C')]")
start=s.index('diagnostics={}');end=s.index('support=',start)
s=s[:start]+'''diagnostics={}
for name,meta in v['diagnostic_files'].items():
 p=D/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'];rows=json.loads(p.read_text(encoding='utf8'));assert len(rows)==meta['rows'];arm,stage=p.stem.split('_');stage=int(stage)
 assert arm in 'CRG' and len(rows)=={7:477,8:899,9:1329}[stage]
 scored=[]
 for row in rows:
  c=np.asarray(row['student_component_group_costs'],dtype=float).sum(-1)
  d=np.asarray(row['student_component_anchor_group_distances'],dtype=float).sum(-1)
  tc=np.asarray(row['teacher_component_group_costs'],dtype=float).sum(-1)[row['teacher_mode']]
  z=np.asarray(row['student_logits'],dtype=float);k=int(z.argmax());assert k==row['student_mode']
  w=row['quality_weight'];tau=v['teacher_audits'][p.stem]['tau'];assert abs(w-tau/(tau+row['frozen_teacher_cost']))<1e-7
  dot=(d+tc-c)/2;alpha=np.clip(dot/max(tc,1e-300),0,1)
  anchor=d-2*alpha*dot+alpha**2*tc
  assert np.allclose(anchor,row['directional_distances'],atol=1e-9,rtol=1e-9)
  assert np.allclose(alpha,row['projection_fractions'],atol=1e-9,rtol=1e-9)
  assert np.all(anchor>=-1e-9) and np.all(anchor<=c+1e-8)
  score=(c+w*d)/(1+w) if arm=='R' else c
  choices=np.flatnonzero(score==score.min())
  best=int(choices[np.argmax(z[choices])]) if arm=='R' else int(score.argmin())
  hinge=max(0.,.1+float(np.max(np.delete(z,best)))-float(z[best]))
  ex=float(score[k]);keep=float(w*anchor[k]) if arm=='G' else 0.
  assert np.isfinite([ex,keep,hinge,*c,*d,*anchor]).all()
  scored.append(dict(index=row['index'],task=row['task'],expert_exec=float(c[k]),expert_best=float(c.min()),teacher_distance_exec=float(d[k]),directional_exec=float(anchor[k]),projection_fraction=float(alpha[k]),score_exec=ex,score_best=float(score.min()),regret=float(c[k]-c.min()),margin_route=hinge,weighted_anchor=keep,auxiliary_loss=ex+keep+.1*hinge))
 diagnostics[p.stem]=dict(count=len(scored),mean_expert_exec=float(np.mean([r['expert_exec'] for r in scored])),p90_expert_exec=float(np.percentile([r['expert_exec'] for r in scored],90)),mean_regret=float(np.mean([r['regret'] for r in scored])),rows=scored)
 (D/('RECOMPUTED_'+p.stem+'.json')).write_text(json.dumps(diagnostics[p.stem],indent=2),encoding='utf8')
assert len(diagnostics)==9 and v['diagnostic_calls']==516 and v['diagnostic_windows']==16230
''' +s[end:]
start=s.index('support=');end=s.index('out=dict(',start)
s=s[:start]+'''support=(metrics['G']['OldFinal']['percent']>metrics['C']['OldFinal']['percent'] and matrix['G']['9_8']['successes']>=matrix['C']['9_8']['successes'] and matrix['G']['9_9']['successes']>=matrix['C']['9_9']['successes'])
side=[t for t in [6,7,8] if matrix['G']['9_'+str(t)]['successes']<matrix['C']['9_'+str(t)]['successes']]
''' +s[end:]
s=s.replace('epochs=1600,updates=304400,own_isolated_optimizer_updates=70,formal_extra_autograd_calls=192,admission_extra_autograd_calls=48','epochs=1200,updates=228300,own_isolated_optimizer_updates=60,formal_extra_autograd_calls=156,admission_extra_autograd_calls=39')
s=s.replace('episodes=800,vectors=80,cells=40','episodes=600,vectors=60,cells=30').replace('strict_CPs=16','strict_CPs=12')
s=s.replace('CCBP-v2','CCBP-v3').replace('V-C、V-N','G-C、G-R、R-C')
s=s.replace('1600epoch304400主更新，70隔离准入更新、正式192额外分项autograd及准入48次另列；16CP、800回合80vector40cell','1200epoch228300主更新，60隔离准入更新、正式156额外分项autograd及准入39次另列；12CP、600回合60vector30cell')
s=s.replace("'no uniform-anchor ablation; cannot isolate quality weighting against fixed weights'", "'R-to-G changes geometry, scaling and routing score; cannot attribute only to geometry; rolling teachers are endogenous'" )
# Final paths must also validate event sequence / all 60 complete batches.
needle="light={n:json.loads"
pos=s.index(needle)
s=s[:pos]+"subprocess.run([sys.executable,str(R/'audit_evaluation_progress.py'),str(D)],check=True)\nev=json.loads((D/'EVALUATION_PROGRESS_AUDIT.json').read_text(encoding='utf8'));assert ev['completed_vectors']==60 and ev['closed_cells']==30 and not ev['ongoing']\n"+s[pos:]
ast.parse(s);(R/'audit_final_local.py').write_text(s,encoding='utf8')
print('Final CPU/local paths AST-checked, not executed and not accepted.')
