"""Describe the fixed CCBP rule on existing accepted scalars; no threshold search."""
import collections,hashlib,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'final_readonly_20261024_224310'
v=json.loads((D/'EVIDENCE.json').read_text());audit=json.loads((D/'LOCAL_FINAL_AUDIT.json').read_text())
assert audit['status']=='COMPLETE_LOCAL_INDEPENDENT_PASS' and audit['final_evidence_sha256']==hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest()
result={}
for arm,rows in v['diagnostics'].items():
 cardinality=collections.Counter();accepted_worse=[];rejected=[];teachers=[];slacks=[];selected_good=0
 for r in rows:
  teacher=sum(r['teacher_component_group_costs'][r['teacher_mode']]);costs=[sum(g) for g in r['student_component_group_costs']];selected=costs[r['student_mode']];budget=teacher+.05
  good=[i for i,c in enumerate(costs) if c<=budget];cardinality[len(good)]+=1;teachers.append(teacher);slacks.append(budget-selected)
  selected_good+=r['student_mode'] in good
  if r['student_mode'] in good and selected>min(costs):accepted_worse.append(dict(sample=r['sample'],cost=selected,minimum=min(costs),teacher=teacher,budget=budget,good_count=len(good)))
  if r['student_mode'] not in good:rejected.append(dict(sample=r['sample'],selected_cost=selected,minimum_cost=min(costs),teacher_cost=teacher,budget=budget,good_count=len(good),selected_mode=r['student_mode'],good_modes=good,group_costs=r['student_component_group_costs'][r['student_mode']]))
 epochs=json.loads(v['light']['EPOCHS_'+arm+'.json']['text'])
 summary={}
 if arm!='A':
  for name,part in [('all',epochs),('first10',epochs[:10]),('last10',epochs[-10:])]:
   count=sum(r['aux_totals']['count'] for r in part)
   summary[name]=dict(sample_visits=count,mean_expert_cost=sum(r['aux_totals']['expert_selected_sum'] for r in part)/count,mean_budget=sum(r['aux_totals']['teacher_budget_sum'] for r in part)/count,mean_excess=sum(r['aux_totals']['excess_sum'] for r in part)/count,mean_route_hinge=sum(r['aux_totals']['route_sum'] for r in part)/count,empty_fallback_visits=sum(r['aux_totals']['empty'] for r in part),all_good_visits=sum(r['aux_totals']['all_good'] for r in part))
 result[arm]=dict(fixed_epsilon=.05,accepted_set_cardinality=dict(sorted(cardinality.items())),selected_accepted_count=selected_good,selected_rejected=rejected,accepted_but_not_minimum=accepted_worse,teacher_cost_median=statistics.median(teachers),teacher_budget_median=statistics.median(teachers)+.05,selected_cost_median=statistics.median(sum(r['student_component_group_costs'][r['student_mode']]) for r in rows),slack_median=statistics.median(slacks),online_epoch_statistics=summary)
out=dict(status='PASS_LOCAL_FIXED_RULE_RECOMPUTATION',source_evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),source_audit_sha256=hashlib.sha256((D/'LOCAL_FINAL_AUDIT.json').read_bytes()).hexdigest(),source_code_sha256={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['losses.py','train.py']},arms=result,limitations=['Final stored scalars have component costs but no logits; cannot reconstruct final routing margin or probabilities.', 'Epoch statistics are aggregated online before different updates, not final-model diagnostics.', 'No per-loss gradient norms or gradient conflict measurements.', 'No trajectory-state causal intervention or held-out risk calibration.'],new_forward_calls=0,new_optimizer_updates=0,new_environments=0,new_thresholds_tested=0)
(D/'BUDGET_MECHANISM_ANALYSIS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
