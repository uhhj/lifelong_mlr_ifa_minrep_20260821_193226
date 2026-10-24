"""Full local matrix/weights/score audit from external CPU evidence; no new rollouts."""
import sys,json,hashlib,statistics,math
from pathlib import Path
import numpy as np
D=Path(sys.argv[1]);R=Path(__file__).resolve().parent;v=json.loads((D/'EVIDENCE.json').read_text());assert v['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and v['no_residual'];light={}
for n,x in v['light'].items():
 assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'];light[n]=json.loads(x['text']) if n.endswith('.json') else x['text']
assert light['controller.exit']==light['evaluate.exit']=='0';parent=R.parent/'20261022_align_object_extension_goal_conditional/final_GOAL_41_20261023_052225/current/ra_coreset/dataset_manifests/task9.json'
assert hashlib.sha256(parent.read_bytes()).hexdigest()==v['manifest_sha256'];manifest=json.loads(parent.read_text())['rows'];assert len(manifest)==3365
init=light['INITIAL_B.json'];counts={str(t):dict(population=sum(r['task']==t for r in manifest),valid=sum(r['task']==t and r['bc_valid'] for r in manifest)) for t in [6,7,8,9]};assert counts==v['dataset_counts']
for arm in 'BCNV':
 assert light['INITIAL_'+arm+'.json']==init and light['ADMISSION_'+arm+'.json']['status']=='PASS';rows=light['EPOCHS_'+arm+'.json'];assert len(rows)==100
 assert all(r['epoch']==i and r['steps']==337*(i+1) and r['totals']['valid']==3218 and r['totals']['invalid']==147 and r['totals']['old_eligible']==430 for i,r in enumerate(rows))
 assert all(r['sampler_sha256']==z['sampler_sha256'] for r,z in zip(rows,light['EPOCHS_B.json']))
 assert light['train_'+arm+'.exit']=='0' and v['strict'][arm]['strict_keys']==100
 grads=light['GRADIENTS_'+arm+'.json'];assert [r['epoch'] for r in grads]==[0,9,49,99] and all(r['extra_autograd_calls']==4 for r in grads)
 for r in grads:
  assert all(math.isfinite(r[k]['l2']) and r[k]['l2']>=0 for k in ['bc','ifa_weighted','exec_weighted','route_weighted'])
rec=light['OUTCOMES.json'];lookup={(r['arm'],r['task'],r['initial_index']):r for r in rec};assert len(rec)==len(lookup)==320
assert set(lookup)=={(a,t,i) for a in 'BCNV' for t in [6,7,8,9] for i in range(20)}
for r in rec:
 assert r['attempt']==1 and 0<r['steps']<=600 and (r['success'] or r['steps']==600) and r['env_seed']==41+r['initial_index']//10
 assert r['cp_sha256']==v['strict'][r['arm']]['cp_sha256']
 noise=np.random.RandomState(20261024+41000+r['task']*100+r['initial_index']).standard_normal((600,7)).astype(np.float32);assert hashlib.sha256(noise.tobytes()).hexdigest()==r['noise_sha256']
 ref=lookup['B',r['task'],r['initial_index']];assert all(r[k]==ref[k] for k in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed'])
events=[json.loads(x) for x in light['EVAL_EVENTS.jsonl'].splitlines()];runs={f'{a}_t{t}_g{g}' for a in 'BCNV' for t in [6,7,8,9] for g in [0,1]}
for kind in ['start','complete']:
 part=[r for r in events if r['kind']==kind];assert len(part)==32 and {r['run'] for r in part}==runs
matrix={a:{str(t):sum(lookup[a,t,i]['success'] for i in range(20)) for t in [6,7,8,9]} for a in 'BCNV'};paired={}
for c in 'BCN':
 paired[c]={str(t):dict(SF=[i for i in range(20) if lookup[c,t,i]['success'] and not lookup['V',t,i]['success']],FS=[i for i in range(20) if not lookup[c,t,i]['success'] and lookup['V',t,i]['success']]) for t in [6,7,8,9]}
weights=v['weights'];tau=max(1e-8,statistics.median(r['teacher_cost'] for r in weights['rows']));assert abs(tau-weights['tau'])<1e-14;wm={r['sample_sha256']:r for r in weights['rows']};assert len(wm)==430
diag={}
for arm,rows in v['diagnostics'].items():
 assert len(rows)==430 and {r['sample_sha256'] for r in rows}==set(wm);costs=[];scores=[];regrets=[];routing=[]
 for r in rows:
  c=np.array(r['student_component_group_costs']).sum(-1);d=np.array(r['student_component_anchor_group_distances']).sum(-1);w=wm[r['sample_sha256']]['weight'];assert abs(w-tau/(tau+wm[r['sample_sha256']]['teacher_cost']))<1e-7
  tc=sum(r['teacher_component_group_costs'][r['teacher_mode']]);assert abs(tc-wm[r['sample_sha256']]['teacher_cost'])<1e-5
  j=(c+w*d)/(1+w) if arm=='V' else c if arm in 'CN' else d;z=np.array(r['student_logits']);k=int(z.argmax());assert k==r['student_mode'];best=j.min();zr=z[j==best].max()
  costs.append(float(c[k]));scores.append(float(j[k]));regrets.append(float(j[k]-best));routing.append(float(((j-best)*np.logaddexp(0,z-zr)).sum()/4))
 diag[arm]=dict(expert_cost_mean=statistics.mean(costs),own_score_mean=statistics.mean(scores),own_score_regret_mean=statistics.mean(regrets),own_score_regret_max=max(regrets),gap_weighted_route_diagnostic_mean=statistics.mean(routing),note='For B/C this new routing quantity is descriptive, not their optimized legacy hinge.')
support=matrix['V']['8']>matrix['C']['8'] and matrix['V']['9']>=matrix['C']['9'];side_effects={t:matrix['V'][t]-matrix['C'][t] for t in ['6','7']}
out=dict(status='COMPLETE_LOCAL_INDEPENDENT_PASS',matrix_success_out_of_20=matrix,paired_V_vs_controls=paired,primary_exploratory_support=support,other_old_task_differences_vs_C=side_effects,all_other_old_tasks_noninferior_observed=all(x>=0 for x in side_effects.values()),diagnostics=diag,teacher_tau=tau,formal_epochs=400,main_updates=134800,own_admission_updates=18,formal_gradient_extra_autograd_calls=64,episodes=320,vectors=32,cells=16,automatic_expansion=False,contract=v['contract'],evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),limitations=['exploratory previously used seed/initials','no uniform-anchor ablation','proxy not calibrated closed-loop risk','no hard routing guarantee','extra autograd/forwards not equal compute'])
(D/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
