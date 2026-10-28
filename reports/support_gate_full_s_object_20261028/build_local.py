"""Compose a new full-sequence method from reviewed full-S and gated-fork code.
Only writes this new local directory; never deploys or trains.
"""
from pathlib import Path
import ast, json, hashlib, difflib, sys
R=Path(__file__).resolve().parent
T=R.parent/'20261026_ccbp_v4_s_object41_42_43'
M=R.parent/'20261027_support_gate_forks'
NAME='SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1'
OLD='CCBP_V4_S_OBJECT41_42_43_20261026_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
def put(p,s):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8')
prepath=Path(sys.argv[1])/'EVIDENCE.json';pre=js(prepath)
assert pre['status']=='PASS_PREDEPLOY_READONLY'
put(R/'PREFLIGHT_LOCAL_AUDIT.json',json.dumps(dict(status=pre['status'],observed=pre['observed'],raw_path=str(prepath),sha256=sha(prepath),capacity=pre['capacity'],hardware=pre['hardware'],new_scientific_calls=0),indent=2)+'\n')
bundle={};review={}
for seed in (41,42,43):
 parent=pre['parents'][str(seed)];source=T/('seed%d'%seed);mapping={}
 for n in ['layout.py','runtime.py','legacy_losses.py','losses.py','memory.py','initialization.py','train.py','collect.py','evaluate.py','synthetic_checks.py']:
  mapping[n]=(source/n).read_text(encoding='utf8')
 mapping['layout.py']=mapping['layout.py'].replace(OLD,NAME).replace("'/data':6*1024**3","'/data':8*1024**3")
 for key,field in [('POP','population'),('VALID','valid'),('OLD','old_valid')]:
  line=next(x for x in mapping['layout.py'].splitlines() if x.startswith(key+'='))
  assert ast.literal_eval(line.split('=',1)[1])=={int(k):v[field] for k,v in parent['stages'].items()}
 assert repr(parent['output']) in mapping['runtime.py'] and repr(parent['source']) in mapping['runtime.py']
 mapping['legacy_memory.py']=mapping['memory.py']
 for n in ['updates.py','legacy_updates.py','support_memory.py','fork_tools.py']:
  mapping[n]=(M/'OBJECT_41'/n).read_text(encoding='utf8')
 s=mapping['support_memory.py'];start=s.index('def load_memory(');end=s.index('\ndef weights(',start)
 s=s[:start]+'''def load_memory(arm,stage):
    from memory import load_memory as load_acquired
    out=load_acquired(arm,stage)
    for task,x in out.items():
        meta=js(R/('MEMORY_%s_%d.json'%(arm,task)))
        ids=set(int(row['trajectory']) for row in x['rows'])
        assert meta['training_attempts']==10 and len(ids)==meta['collection_successes']
        assert all(row['source']=='online' for row in x['rows'])
        assert all(0<=i<10 for i in ids)
        x['support']=len(ids)/10.;x['support_trajectories']=sorted(ids)
    return out
''' +s[end:];mapping['support_memory.py']=s
 s=mapping['fork_tools.py'].replace('STAGE','stage')
 s=s.replace('def setup_probes(ds,memory):','def setup_probes(ds,memory):\n    stage=ds.stage')
 s=s.replace('def directional_diagnostic(a,batch,memory,pr,arm):','def directional_diagnostic(a,batch,memory,pr,arm):\n    stage=pr[\'ds\'].stage')
 mapping['fork_tools.py']=s
 mapping['gate_admission.py']=(M/'train_fork.py').read_text(encoding='utf8').split('\ndef main():')[0]+'\n'
 s=mapping['train.py'].replace('from memory import load_memory,slots,retention','from memory import slots,retention\nfrom support_memory import load_memory\nfrom fork_tools import setup_probes,probe,choose,directional_diagnostic\nimport gate_admission')
 s=s.replace('history=[];gradients=[]','history=[];gradients=[];gates=[];physical=0')
 s=s.replace("count=len(memory[t]['rows']))", "count=len(memory[t]['rows']),support=memory[t]['support'],support_trajectories=memory[t]['support_trajectories'])")
 s=s.replace('            admit(arm,stage,ds,a,memory,scale,info)', '''            pr=setup_probes(ds,memory) if stage>6 else None
            if stage==6:
                admit(arm,stage,ds,a,memory,scale,info)
            else:
                write(R/('PROBES_%d.json'%stage),pr['metadata'])
                gate_admission.STAGE=stage
                own=gate_admission.admission(a,ds,memory,scale,info,pr,'G')
                assert own['optimizer_calls']==29 and own['extra_autograd_calls']==11
                write(R/('ADMISSION_%s_%d.json'%(arm,stage)),dict(arm=arm,stage=stage,method='support_gate_full_S',contract=verify(),**own))''')
 start=s.index('                for bi,batch in enumerate(loader):');end=s.index("                assert tot['valid']",start)
 s=s[:start]+'''                iterator=iter(loader);bi=0;retained_steps=0
                while True:
                    block=[]
                    for _ in range(10 if stage>6 else 1):
                        try:block.append(next(iterator))
                        except StopIteration:break
                    if not block:break
                    if stage>6 and ep in [0,9,49,99] and not measured:
                        dg=directional_diagnostic(a,block[0],memory,pr,'Q')
                        if dg is not None:
                            measured=True;gradients.append(dict(stage=stage,epoch=ep,batch=bi,steps=a.step,**dg));write(R/('GRADIENTS_'+arm+'.json'),gradients)
                    if stage>6:
                        base=capture(a)
                        crows=[update(a,b,'C',memory,scale,info,stage,retain_weight=0.) for b in block]
                        cstate=capture(a);cv=probe(a,pr,memory)
                        restore(a,base)
                        qrows=[update(a,b,'Q',memory,scale,info,stage) for b in block]
                        qstate=capture(a);qv=probe(a,pr,memory)
                        decision,checks=choose(cv,qv);selected=cstate if decision=='C' else qstate
                        restore(a,selected);assert equal(selected,capture(a))
                        rows=crows if decision=='C' else qrows
                        gate=dict(stage=stage,epoch=ep,batch=bi,steps=len(block),start_step=base['step'],end_step=a.step,C=cv,Q=qv,accepted=decision,checks=checks)
                        gates.append(gate)
                        with (R/'GATES.jsonl').open('a') as f:f.write(json.dumps(gate)+'\\n')
                        if decision=='Q':retained_steps+=len(block)
                        physical+=2*len(block)
                        del base,cstate,qstate,selected
                    else:
                        rows=[update(a,block[0],arm,memory,scale,info,stage)]
                        physical+=1
                    for st in rows:
                        for k in tot:tot[k]+=st[k]
                    bi+=len(block)
                assert bi==len(loader)
''' +s[end:]
 s=s.replace("(4*len(loader) if stage>6 else 0)","4*retained_steps")
 s=s.replace("totals=tot)","totals=tot,physical_optimizer_calls=physical,gate_blocks=len(gates),gate_Q_accepts=sum(g['accepted']=='Q' for g in gates),training_probe=probe(a,pr,memory) if pr else None)")
 s=s.replace('history=history))','history=history,physical_optimizer_calls=physical,probe_sha256=sha(R/(\'PROBES_%d.json\'%stage)) if pr else None,gate_blocks=len(gates)))')
 s=s.replace("phase='train',epoch_complete=ep+1,steps=a.step", "phase='train',epoch_complete=ep+1,steps=a.step,physical_optimizer_calls=physical")
 s=s.replace('assert a.step==105500 and len(history)==400','assert a.step==105500 and len(history)==400 and physical==195300 and len(gates)==9100')
 s=s.replace("status='PASS',epochs=400,steps=a.step", "status='PASS_EXECUTION_PENDING_FINAL',epochs=400,steps=a.step,physical_optimizer_calls=physical,gate_blocks=len(gates),Q_accepts=sum(g['accepted']=='Q' for g in gates)")
 mapping['train.py']=s
 for n,s in list(mapping.items()):
  ast.parse(s,feature_version=(3,8))
  original=(source/n).read_text(encoding='utf8') if (source/n).exists() else (M/'OBJECT_41'/n).read_text(encoding='utf8') if (M/'OBJECT_41'/n).exists() else ''
  review[f'seed{seed}/{n}']=dict(source_sha256=hashlib.sha256(original.encode()).hexdigest(),new_sha256=hashlib.sha256(s.encode()).hexdigest(),unchanged=s==original)
  if s!=original:put(R/'review'/f'seed{seed}_{n}.diff',''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='reference/'+n,tofile=f'new/seed{seed}/'+n)))
 for n in ['initialization.py','collect.py','evaluate.py','memory.py','losses.py','legacy_losses.py','synthetic_checks.py','runtime.py']:
  assert mapping[n]==(source/n).read_text(encoding='utf8')
 mapping['INITIAL_SPLIT.json']=json.dumps(parent['initial_split'],indent=2)+'\n'
 mapping['PARENT_BINDING.json']=json.dumps(parent,indent=2)+'\n'
 mapping['PROTOCOL.md']=(R/'PROTOCOL.md').read_text(encoding='utf8')
 for n,s in mapping.items():put(R/f'seed{seed}'/n,s);bundle[f'seed{seed}/{n}']=s
controller=(T/'controller.py').read_text(encoding='utf8').replace('CCBP_V4_S_OBJECT41_42_43_20261026.lock',NAME+'.lock')
put(R/'controller.py',controller)
for n in ['controller.py','PROTOCOL.md']:bundle[n]=(R/n).read_text(encoding='utf8')
bundle['PREDEPLOY.json']=json.dumps(dict(observed=pre['observed'],evidence_sha256=sha(prepath),seed41_contract=pre['seed41_contract'],capacity=pre['capacity'],hardware=pre['hardware'],source_review=review),indent=2)+'\n'
put(R/'DEPLOY_TEXT_BUNDLE.json',json.dumps(bundle,ensure_ascii=True,indent=2)+'\n')
put(R/'UPLOAD_REVIEW.json',json.dumps(dict(status='REVIEWED_TEXT_ONLY_PENDING_REMOTE_IMPORT_GPU_ADMISSION',files={n:dict(bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()) for n,s in bundle.items()},source_review=review,private_payloads_included=False),indent=2)+'\n')
deploy=(T/'deploy_remote.py').read_text(encoding='utf8').replace(OLD,NAME).replace("('/data',6*1024**3","('/data',8*1024**3")
deploy=deploy.replace("method='S',budget=dict(epochs=400,updates=105500,collection_episodes=40,formal_episodes=200)","method='S_support_q_C_reference_gate_10',budget=dict(epochs=400,updates=105500,physical_optimizer_calls=195300,C_candidate_calls=89800,admission_optimizer_calls=91,admission_extra_autograd=41,collection_episodes=40,formal_episodes=200)")
deploy=deploy.replace('import runtime,memory,updates,initialization,train,collect,evaluate;','import runtime,memory,support_memory,fork_tools,gate_admission,updates,initialization,train,collect,evaluate;')
deploy=deploy.replace('updates=316500,training_episodes=120','updates=316500,physical_optimizer_calls=585900,C_candidate_calls=269400,admission_optimizer_calls=273,admission_extra_autograd=123,training_episodes=120')
deploy=deploy.replace('assert not Q.exists()',"stop=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FORKS_20261027_v1/user_stop_20261028/RESULT.json')\nassert json.loads(stop.read_text())['status']=='USER_STOPPED_PARTIAL_PRESERVED_NOT_SCIENTIFIC_ACCEPTANCE'\nassert not Q.exists()")
ast.parse(deploy,feature_version=(3,8));put(R/'deploy_remote.py',deploy)
put(R/'publish.py',(M/'publish.py').read_text(encoding='utf8').replace('reports/support_gate_forks_20261027','reports/support_gate_full_s_object_20261028').replace('Support-calibrated retention finite common-start forks','Latest complete support-gated S OBJECT three seeds'))
print(json.dumps(dict(status='BUILT_NOT_DISPATCHED',files=len(bundle),budget_physical=585900,review=str(R/'UPLOAD_REVIEW.json'))))
