"""External read-only audit adaptation, no scientific execution."""
from pathlib import Path
import ast,json,hashlib
R=Path(__file__).resolve().parent;OLD=R.parent/'20261027_c_goal42_43_object41_42_43'
NAME='C_MATCH_FULL_S_BUDGET_GOAL_OBJECT_20261030_v1'
CASES=[s+'_'+str(k) for s in ['GOAL','OBJECT'] for k in [41,42,43]]
def read(p):return p.read_text(encoding='utf8')
def put(p,s):ast.parse(s,feature_version=(3,8));p.write_text(s,encoding='utf8')
def rep(s,a,b):assert a in s,a;return s.replace(a,b)
s=read(R.parent/'20261029_support_gate_full_s_goal/check_readonly.py').replace('SUPPORT_GATE_FULL_S_GOAL41_42_43_20261029_v1',NAME).replace("for seed in (41,42,43):\n q=Q/('seed%d'%seed);cc=json.loads((q/'CONTRACT.json').read_text());assert sha(q/'CONTRACT.json')==c['children'][str(seed)]", "for case in "+repr(CASES)+":\n q=Q/case;cc=json.loads((q/'CONTRACT.json').read_text());assert sha(q/'CONTRACT.json')==c['children'][case]")
s=s.replace("('/data',8*1024**3","('/data',18*1024**3")
put(R/'check_readonly.py',s)
s=read(OLD/'audit_progress.py')
s=rep(s,"cases=['GOAL_42','GOAL_43','OBJECT_41','OBJECT_42','OBJECT_43']",'cases='+repr(CASES))
s=rep(s,"  t=6+i//100;st=stages[str(t)];steps+=st['batches']", "  t=6 if i<100 else 7+(i-100)//200;ep=i if i<100 else (i-100)%200;st=stages[str(t)];steps+=st['batches']")
s=s.replace("==(t,i%100,st['batches'],steps)","==(t,ep,st['batches'],steps)")
s=rep(s," counts[case]=dict(","  assert row['physical_optimizer_calls']==steps\n  assert row['isolated_optimizer_calls']==4+29*(t-6)\n  schedule=get('SCHEDULE_C_%d.json'%t);assert schedule['total_steps']==st['batches']*(100 if t==6 else 200) and schedule['warmup_steps']==500\n  key=str(sum(stages[str(j)]['batches']*(100 if j==6 else 200) for j in range(6,t)))\n  assert get('COPY_EQ.json')['checked'][key]['all_state_exact']\n counts[case]=dict(")
s=s.replace("(4 if t==6 else 5)","(4 if t==6 else 29)")
s=s.replace("if t>6:assert a['stateful_gradient_measurement_exact']","if t>6:assert a['stateful_gradient_measurement_exact'] and a['ten_step_and_checkpoint_continuation_exact']")
s=s.replace("sm['epochs']==100", "sm['epochs']==(100 if t==6 else 200)")
s=s.replace("stages[str(j)]['batches']*100", "stages[str(j)]['batches']*(100 if j==6 else 200)")
s=s.replace("len(rows)==400 and steps==sum(st['batches']*100 for st in stages.values())", "len(rows)==700 and steps==sum(st['batches']*(100 if int(t)==6 else 200) for t,st in stages.items())")
s=rep(s,"out=dict(status=",'''prefix_verified=None
if len(sys.argv)>2:
 previous=js(Path(sys.argv[2])/'EVIDENCE.json')
 assert previous['contract_sha256']==v['contract_sha256']
 for n,x in previous['light'].items():
  if n.endswith(('EPOCHS_C.json','OUTCOMES.json','GRADIENTS_C.json')):
   old=json.loads(x['text']);assert L[n][:len(old)]==old,('prefix',n)
  elif n.endswith('EVAL_EVENTS.jsonl'):assert L[n].startswith(x['text']),('prefix',n)
 prefix_verified=str(sys.argv[2])
out=dict(prefix_verified=prefix_verified,status=''' )
s=s.replace("source_evidence_sha256=sha(D/'EVIDENCE.json')", "source_evidence_sha256=sha(D/'EVIDENCE.json'),snapshot_sha256=sha(D/'EVIDENCE.json')")
put(R/'audit_progress.py',s)

# Retain real CPU checkpoint/teacher/initial-byte verification; update only budget/indexing.
tree=ast.parse(read(OLD/'final_readonly.py'));old_scripts=ast.literal_eval(tree.body[0].value)
scripts={}
for case in CASES:
 oldcase=case if case!='GOAL_41' else 'GOAL_42';s=old_scripts[oldcase]
 parent=json.loads(read(R/case/'PARENT_BINDING.json'));oldparent=json.loads(read(OLD/oldcase/'PARENT_BINDING.json'))
 s=s.replace('CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1/'+oldcase,NAME+'/'+case)
 s=s.replace(repr(oldparent['output']),repr(parent['output'])).replace(repr(oldparent['source']),repr(parent['source']))
 s=s.replace('len(hist)==400','len(hist)==700')
 s=rep(s,"  stage=6+i//100;order=torch.randperm(POP[stage],generator=torch.Generator().manual_seed(i%100))", "  stage=6 if i<100 else 7+(i-100)//200;ep=i if i<100 else (i-100)%200;order=torch.randperm(POP[stage],generator=torch.Generator().manual_seed(ep))")
 s=s.replace("rec['next_epoch']==100", "rec['next_epoch']==200")
 oldsteps=76100 if case.startswith('GOAL') else 105500
 s=s.replace("rec['steps']==%d"%oldsteps,"rec['steps']==TOTAL_UPDATES").replace("rec['state']['step']==%d"%oldsteps,"rec['state']['step']==TOTAL_UPDATES")
 s=rep(s," assert rec['manifest_sha256']==", " assert rec['physical_calls']=={'production':TOTAL_UPDATES,'admission':91}\n assert rec['state']['scheduler']['last_epoch']==STAGE_EPOCHS[9]*((POP[9]+9)//10)\n assert rec['manifest_sha256']==")
 s=rep(s,"assert all(v==task6['C'] for v in task6.values())", "assert all(v==task6['C'] for v in task6.values())\nsref=js(Q/'S_REFERENCE.json');assert all(sha(x['path'])==x['sha256'] for x in sref.values())\nss=torch.load(sref['stage6_checkpoint']['path'],map_location='cpu')['state_dict'];assert {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in ss.items()}==task6['C']")
 ast.parse(s,feature_version=(3,8));scripts[case]=s
 put(R/('final_'+case+'_readonly.py'),s)
tail=read(OLD/'final_readonly.py').split('"""Read-only aggregate final CPU audits;',1)[1]
tail='"""Read-only aggregate final CPU audits;'+tail
tail=tail.replace('CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1',NAME)
put(R/'final_readonly.py','SCRIPTS='+repr(scripts)+'\n'+tail)
s=read(OLD/'audit_final_local.py').replace("a['formal_episodes']==1000 and a['isolated_optimizer_calls']==95", "a['formal_episodes']==1200 and a['isolated_optimizer_calls']==546")
s=s.replace('==2000','==4200').replace('==468700','==1012500')
s=s.replace("steps=sum(st['batches']*100 for st in stages.values())", "steps=sum(st['batches']*(100 if int(t)==6 else 200) for t,st in stages.items())")
s=s.replace("['epochs']==400","['epochs']==700").replace('[0,9,49,99]','[0,19,99,199]')
s=s.replace('epochs=400,isolated_updates=19','epochs=700,isolated_updates=91')
s=s.replace('epochs=2000,updates=468700,isolated_updates=95,formal_extra_autograd=240,admission_extra_autograd=60','epochs=4200,updates=1012500,isolated_updates=546,formal_extra_autograd=288,admission_extra_autograd=72')
s=s.replace('episodes=1000,vectors=100,cells=50','episodes=1200,vectors=120,cells=60').replace('strict_CPs=20,complete_recoveries=5,teacher_caches=15','strict_CPs=24,complete_recoveries=6,teacher_caches=18')
put(R/'audit_final_local.py',s)
review=dict(status='STATIC_ONLY_NOT_ACTUAL_FINAL_ACCEPTANCE',new_C_runs=6,new_S_runs=0,production_calls=1012500,isolated_calls=546,formal=1200,CPs=24,recoveries=6,teacher_caches=18,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*readonly.py')})
(R/'FINAL_WRAPPER_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps(review))
