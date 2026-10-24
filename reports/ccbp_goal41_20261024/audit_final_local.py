"""Recompute complete matrices, paired outcomes and cost diagnostic from scalar evidence."""
import sys,json,hashlib,statistics,math
from pathlib import Path
D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text());assert v['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and v['no_residual'];light={}
for n,x in v['light'].items():
 assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'];light[n]=json.loads(x['text']) if n.endswith('.json') else x['text']
assert light['controller.exit']=='0' and light['evaluate.exit']=='0'
counts=v['dataset_counts'];valid=sum(x['valid'] for x in counts.values());initial=light['INITIAL_A.json']
parent=Path(__file__).resolve().parent.parent/'20261022_align_object_extension_goal_conditional/final_GOAL_41_20261023_052225/current/ra_coreset/dataset_manifests/task9.json'
assert hashlib.sha256(parent.read_bytes()).hexdigest()==v['manifest_sha256']
parent_rows=json.loads(parent.read_text())['rows']
expected_counts={str(t):dict(population=sum(r['task']==t for r in parent_rows),valid=sum(r['task']==t and r['bc_valid'] for r in parent_rows)) for t in [6,7,8,9]}
assert counts==expected_counts and valid==3218 and counts['8']['valid']==430
for arm in 'ABCD':
 assert light['INITIAL_'+arm+'.json']==initial
 assert light['ADMISSION_'+arm+'.json']['status']=='PASS'
 rows=light['EPOCHS_'+arm+'.json'];assert len(rows)==100
 assert all(r['epoch']==i and r['steps']==337*(i+1) and r['totals']['valid']==valid and r['totals']['invalid']==3365-valid and r['totals']['old_eligible']==counts['8']['valid'] for i,r in enumerate(rows))
 assert all(r['sampler_sha256']==z['sampler_sha256'] for r,z in zip(rows,light['EPOCHS_A.json']))
 assert light['train_'+arm+'.exit']=='0' and v['strict'][arm]['strict_keys']==100
rec=light['OUTCOMES.json'];lookup={(r['arm'],r['task'],r['initial_index']):r for r in rec};assert len(rec)==len(lookup)==160
assert set(lookup)=={(a,t,i) for a in 'ABCD' for t in [8,9] for i in range(20)}
for r in rec:
 assert r['attempt']==1 and 0<r['steps']<=600 and (r['success'] or r['steps']==600)
 assert r['cp_sha256']==v['strict'][r['arm']]['cp_sha256']==light['TRAIN_'+r['arm']+'.json']['cp_sha256']
events=[json.loads(x) for x in light['EVAL_EVENTS.jsonl'].splitlines() if x.strip()]
expected_runs={f'{a}_t{t}_g{g}' for a in 'ABCD' for t in [8,9] for g in [0,1]}
for kind in ['start','complete']:
 part=[r for r in events if r['kind']==kind];assert len(part)==16 and {r['run'] for r in part}==expected_runs
matrix={arm:{str(t):sum(lookup[arm,t,i]['success'] for i in range(20)) for t in [8,9]} for arm in 'ABCD'};paired={}
for arm in 'ABC':
 paired[arm]={}
 for t in [8,9]:
  sf=fs=0
  for i in range(20):
   a=lookup[arm,t,i];d=lookup['D',t,i]
   assert all(a[k]==d[k] for k in ['initial_sha256','noise_sha256','env_seed'])
   sf+=a['success'] and not d['success'];fs+=not a['success'] and d['success']
  paired[arm][str(t)]=dict(control_success_D_failure=sf,control_failure_D_success=fs)
supported=all(matrix['D']['8']>matrix[a]['8'] and matrix['D']['9']>=matrix[a]['9'] for a in 'ABC')
diag={}
for arm,rows in v['diagnostics'].items():
 assert len(rows)==counts['8']['valid'] and len({r['sample'] for r in rows})==len(rows)
 assert {r['sample']:r['sample_sha256'] for r in rows}=={Path(r['path']).name:r['sha256'] for r in parent_rows if r['task']==8 and r['bc_valid']}
 assert all(0<=r['teacher_mode']<5 and 0<=r['student_mode']<5 and all(len(r[k])==5 and all(len(g)==3 and all(math.isfinite(c) and c>=0 for c in g) for g in r[k]) for k in ['teacher_component_group_costs','student_component_group_costs']) for r in rows)
 assert all((r['sample'],r['teacher_mode'],r['teacher_component_group_costs'])==(z['sample'],z['teacher_mode'],z['teacher_component_group_costs']) for r,z in zip(rows,v['diagnostics']['A']))
 teacher=[sum(r['teacher_component_group_costs'][r['teacher_mode']]) for r in rows];student=[sum(r['student_component_group_costs'][r['student_mode']]) for r in rows]
 diag[arm]=dict(count=len(rows),teacher_mean=statistics.mean(teacher),student_mean=statistics.mean(student),excess_mean=statistics.mean(max(0,s-t-.05) for s,t in zip(student,teacher)),excess_count=sum(s>t+.05 for s,t in zip(student,teacher)),improved_count=sum(s<t for s,t in zip(student,teacher)),student_mixture_nll=statistics.mean(r['student_mixture_nll'] for r in rows))
out=dict(status='COMPLETE_LOCAL_INDEPENDENT_PASS',scientific_support=supported,matrix_success_out_of_20=matrix,paired_D_vs_controls=paired,diagnostics=diag,main_updates=134800,admission_updates=sum(light['ADMISSION_'+a+'.json']['optimizer_calls'] for a in 'ABCD'),episodes=160,cells=8,vectors=16,contract=v['contract'],final_evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),automatic_expansion=False,limitations=['one exploratory seed','expert replay cost is not closed-loop risk','coupled noise evaluation differs from original global RNG protocol','B/C/D additional forwards are not equal compute','fixed unoptimized loss weights'])
(D/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
