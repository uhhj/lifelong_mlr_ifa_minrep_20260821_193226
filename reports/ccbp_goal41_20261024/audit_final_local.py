"""Recompute complete matrices, paired outcomes and cost diagnostic from scalar evidence."""
import sys,json,hashlib,statistics
from pathlib import Path
D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text());assert v['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and v['no_residual'];light={}
for n,x in v['light'].items():
 assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'];light[n]=json.loads(x['text']) if n.endswith('.json') else x['text']
assert light['controller.exit']=='0' and light['evaluate.exit']=='0'
counts=v['dataset_counts'];valid=sum(x['valid'] for x in counts.values());initial=light['INITIAL_A.json']
for arm in 'ABCD':
 assert light['INITIAL_'+arm+'.json']==initial
 assert light['ADMISSION_'+arm+'.json']['status']=='PASS'
 rows=light['EPOCHS_'+arm+'.json'];assert len(rows)==100
 assert all(r['epoch']==i and r['steps']==337*(i+1) and r['totals']['valid']==valid and r['totals']['invalid']==3365-valid and r['totals']['old_eligible']==counts['8']['valid'] for i,r in enumerate(rows))
 assert all(r['sampler_sha256']==z['sampler_sha256'] for r,z in zip(rows,light['EPOCHS_A.json']))
 assert light['train_'+arm+'.exit']=='0' and v['strict'][arm]['strict_keys']==100
rec=light['OUTCOMES.json'];lookup={(r['arm'],r['task'],r['initial_index']):r for r in rec};assert len(rec)==len(lookup)==160
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
 teacher=[sum(r['teacher_component_group_costs'][r['teacher_mode']]) for r in rows];student=[sum(r['student_component_group_costs'][r['student_mode']]) for r in rows]
 diag[arm]=dict(count=len(rows),teacher_mean=statistics.mean(teacher),student_mean=statistics.mean(student),excess_mean=statistics.mean(max(0,s-t-.05) for s,t in zip(student,teacher)),excess_count=sum(s>t+.05 for s,t in zip(student,teacher)),improved_count=sum(s<t for s,t in zip(student,teacher)),student_mixture_nll=statistics.mean(r['student_mixture_nll'] for r in rows))
out=dict(status='COMPLETE_LOCAL_INDEPENDENT_PASS',scientific_support=supported,matrix_success_out_of_20=matrix,paired_D_vs_controls=paired,diagnostics=diag,main_updates=134800,admission_updates=sum(light['ADMISSION_'+a+'.json']['optimizer_calls'] for a in 'ABCD'),episodes=160,cells=8,vectors=16,contract=v['contract'],final_evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),automatic_expansion=False,limitations=['one exploratory seed','expert replay cost is not closed-loop risk','coupled noise evaluation differs from original global RNG protocol','B/C/D additional forwards are not equal compute','fixed unoptimized loss weights'])
(D/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
