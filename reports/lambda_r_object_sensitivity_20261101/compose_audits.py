"""External audit wrappers; AST/static preparation is not actual acceptance."""
from pathlib import Path
import json,ast,hashlib
R=Path(__file__).resolve().parent;P=R.parent;S=P/'20261028_support_gate_full_s_object';Q=P/'20261031_full_q_vs_latest_s_two_suites'
NAME='S_LAMBDA_R_OBJECT41_42_43_20261101_v1'
cases=[g+'_'+str(seed) for g in ['L005','L020'] for seed in (41,42,43)]
read=lambda p:p.read_text(encoding='utf8')
def put(p,s):p.write_text(s,encoding='utf8')
def rep(s,a,b):assert a in s,a;return s.replace(a,b)
a=read(S/'audit_progress.py')
a=rep(a,"for seed in (41,42,43):\n prefix='seed%d/'%seed;",f"for case in {cases!r}:\n seed=int(case.split('_')[1]);prefix=case+'/';")
a=a.replace('counts[str(seed)]','counts[case]')
a=rep(a,"  assert row['seconds']>0","  assert row['seconds']>0\n  assert row['physical_calls_observed']==dict(production=phys,admission=4+29*(st-6))")
q=read(Q/'audit_progress.py');start=q.index(" if h:\n  init=js(");end=q.index(' own=ownauto=0',start)
extra=q[start:end].replace('INITIAL_Q','INITIAL_S').replace('SCHEDULE_Q','SCHEDULE_S').replace('TASK6_PARITY_Q','TASK6_PARITY_S').replace("suite=case.split('_')[0]","suite='OBJECT'")
a=rep(a,' own=ownauto=0',extra+' own=ownauto=0')
a=rep(a,"    assert a['status']=='PASS_OWN_GPU_ADMISSION'","    assert a['lambda_R']==p['lambda_R'] and a['uniform_reference']=='same_lambda_uniform_support'\n    assert a['status']=='PASS_OWN_GPU_ADMISSION'")
a=rep(a,"  if '/GATES_stage' in n:","  if n.endswith(('GRADIENTS_S.json','OUTCOMES.json')):\n   before=json.loads(x['text']);assert js(n,[])[:len(before)]==before\n  if '/GATES_stage' in n:")
put(R/'audit_progress.py',a)
put(R/'audit_seed_evaluation.py',read(S/'audit_seed_evaluation.py'))
scripts={}
for case in cases:
 seed=int(case.split('_')[1]);s=read(S/('final_seed%d_readonly.py'%seed))
 s=s.replace('SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1/seed'+str(seed),NAME+'/'+case)
 needle="verify();assert (Q.parent/'controller.exit').read_text()=='0'"
 s=rep(s,needle,needle+"\nassert RETAIN_WEIGHT==js(Q/'PARENT_BINDING.json')['lambda_R']\nassert js(Q/'CONTRACT.json')['lambda_R']==RETAIN_WEIGHT")
 s=rep(s,"   assert rr['physical_optimizer_calls']==physical", "   assert rr['physical_calls_observed']==dict(production=physical,admission=4+29*(stage-6))\n   assert rr['physical_optimizer_calls']==physical")
 # Real CPU reconstruction of the final model and optimizer/scheduler state.
 s=rep(s," assert rec['physical_optimizer_calls']==195300", " optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-4)\n optimizer.load_state_dict(rec['state']['optimizer'])\n sched=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: min(step/500.,max(0.,(100*((POP[9]+9)//10)-step)/(100*((POP[9]+9)//10)-500))))\n sched.load_state_dict(rec['state']['scheduler'])\n optimizer.load_state_dict(rec['state']['optimizer'])\n def eq(x,y):\n  if torch.is_tensor(x):return torch.is_tensor(y) and torch.equal(x,y)\n  if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(eq(x[k],y[k]) for k in x)\n  if isinstance(x,(list,tuple)):return type(x)==type(y) and len(x)==len(y) and all(eq(u,v) for u,v in zip(x,y))\n  return x==y\n assert eq(optimizer.state_dict(),rec['state']['optimizer']) and eq(sched.state_dict(),rec['state']['scheduler'])\n assert rec['physical_optimizer_calls']==195300")
 s=rep(s,"sampling_cursor=rec['retention_sampling_cursor'])","sampling_cursor=rec['retention_sampling_cursor'],CPU_optimizer_scheduler_load_exact=True)")
 scripts[case]=s;put(R/('final_'+case+'_readonly.py'),s)
agg=read(Q/'final_readonly.py');agg=agg[agg.index('"""Only after natural completion:'):]
put(R/'final_readonly.py','CHECK_SCRIPT='+repr(read(R/'check_readonly.py'))+'\nSCRIPTS='+repr(scripts)+'\n'+agg)
a=read(S/'audit_final_local.py')
for old,new in [('a[\'epochs\']==1200',"a['epochs']==2400"),("a['accepted_updates']==316500","a['accepted_updates']==633000"),("a['own_isolated_optimizer_calls']==273","a['own_isolated_optimizer_calls']==546"),("a['own_extra_autograd_calls']==123","a['own_extra_autograd_calls']==246"),("a['physical_training_optimizer_calls']==585900","a['physical_training_optimizer_calls']==1171800"),("a['training_collection_completed_episodes']==120","a['training_collection_completed_episodes']==240"),("a['formal_episodes']==600","a['formal_episodes']==1200")]:a=rep(a,old,new)
a=rep(a,"for seed in (41,42,43):\n key=str(seed);r=dict(v['seeds'][key],light={n[len('seed'+key+'/'):]:x for n,x in v['light'].items() if n.startswith('seed'+key+'/')});p=js(R/f'seed{seed}'/'PARENT_BINDING.json');progress=a['counts'][key]",f"for key in {cases!r}:\n seed=int(key.split('_')[1]);r=dict(v['cases'][key],light={{n[len(key+'/'):]:x for n,x in v['light'].items() if n.startswith(key+'/')}});p=js(R/key/'PARENT_BINDING.json');progress=a['counts'][key]")
a=a.replace("str(child),key]","str(child),str(seed)]")
a=a.replace("f's{seed}_'","key+'_'" )
a=rep(a,"_,kind,arm,st=fp.stem.split('_')","kind,arm,st=fp.stem[len(key)+1:].split('_')")
a=rep(a,"assert r['strict']['S']['recovery']['final_hardlink']","assert r['strict']['S']['recovery']['CPU_optimizer_scheduler_load_exact']\n assert r['strict']['S']['recovery']['final_hardlink']")
a=a.replace('seeds=results,epochs=1200,main_updates=316500,physical_training_optimizer_calls=585900,isolated_optimizer_updates=273,formal_extra_autograd=108,admission_extra_autograd=123,training_collection_episodes=120,formal_episodes=600,vectors=60,cells=30','cases=results,epochs=2400,main_updates=633000,physical_training_optimizer_calls=1171800,isolated_optimizer_updates=546,formal_extra_autograd=216,admission_extra_autograd=246,training_collection_episodes=240,formal_episodes=1200,vectors=120,cells=60')
a=a.replace('Only latest support-gated S OBJECT41/42/43; the canceled mechanism experiment is not resumed or required for acceptance.','Only fixed lambda_R 0.05/0.2 OBJECT S41/42/43; accepted 0.1 read-only. No outcome-based selection.')
put(R/'audit_final_local.py',a)
# Compare new lambda values with accepted lambda=0.1. Q identifiers in the
# template are renamed explicitly to Variant; formulas and paired keys retained.
c=read(Q/'compare_primary_s.py').replace("a['main_updates']==544800","a['main_updates']==633000")
c=rep(c,"for suite,ref in binding['sources'].items():","for suite in ['L005','L020']:\n ref=binding['sources']['OBJECT']")
c=c.replace('INITIAL_Q','INITIAL_S').replace('TASK6_PARITY_Q','TASK6_PARITY_S').replace('EPOCHS_Q','EPOCHS_S').replace('ADMISSION_Q_','ADMISSION_S_')
c=rep(c,"ccall=sum(x['batches'] for x in ch if x['stage']==stage)","ccall=sum(x['batches']*(1 if stage==6 else 2) for x in ch if x['stage']==stage)")
c=rep(c,"ccall*(1 if stage==6 else 2)==scall","ccall==scall")
c=c.replace('Q','Variant').replace('关闭门控','固定系数敏感性').replace('两个套件各三seed','OBJECT两个新系数各三seed')
c=c.replace('Equal accepted updates and collection counts; Variant single candidate uses fewer physical optimizer calls; unequal FLOPs/wall time and own cache contents','Equal accepted and physical optimizer calls and collection counts; own cache contents and wall time can differ')
c=c.replace('S historical engineering diagnostics remain separate; Variant startup v1 zero optimizer and environments independently audited','Baseline 0.1 historical diagnostics remain separate; the baseline 600 outcomes are reused for two paired comparisons, not 1200 independent baseline outcomes')
c=c.replace('Variant单候选不匹配S双候选物理optimizer/FLOPs/墙钟','各档物理optimizer预算相同，FLOPs/墙钟及缓存内容不保证相同')
c=c.replace('EVariantUAL','EQUAL')
c=c.replace('两者各240次训练采集、同配额但各自产生缓存','每档各120次训练采集、同配额但各自产生缓存；两个新系数共240次，已有0.1的120次只读复用')
put(R/'compare_primary_s.py',c)
review={}
for p in R.glob('*.py'):
 ast.parse(read(p),feature_version=(3,8));review[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
put(R/'FINAL_WRAPPER_REVIEW.json',json.dumps(dict(status='STATIC_ONLY_NOT_ACTUAL_ACCEPTANCE',files=review,required_actual=dict(checkpoints=24,recoveries=6,teacher_caches=24,initial_noise_episodes=1200,gate_blocks=54600),actual_CPU_optimizer_or_environment_calls=0),indent=2)+'\n')
print('STATIC_WRAPPERS_PREPARED_NOT_ACCEPTED')
