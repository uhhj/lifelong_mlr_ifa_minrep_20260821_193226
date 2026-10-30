"""Check generated C budget/source bindings against actually accepted S evidence."""
from pathlib import Path
import json,hashlib,ast,datetime
R=Path(__file__).resolve().parent
sources={'GOAL':R.parent/'20261029_support_gate_full_s_goal/final_readonly_20261030_111546','OBJECT':R.parent/'20261028_support_gate_full_s_object/final_readonly_20261029_201235'}
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out={};cases={}
for suite,folder in sources.items():
 a=js(folder/'LOCAL_FINAL_AUDIT.json');e=js(folder/'EVIDENCE.json')
 assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['formal_episodes']==600 and a['epochs']==1200
 assert a['evidence_sha256']==sha(folder/'EVIDENCE.json')
 assert a['physical_training_optimizer_calls']==(426600 if suite=='GOAL' else 585900) and a['isolated_optimizer_updates']==273
 out[suite]=dict(final_path=str(folder.resolve()),evidence_sha256=sha(folder/'EVIDENCE.json'),audit_sha256=sha(folder/'LOCAL_FINAL_AUDIT.json'),physical_calls=a['physical_training_optimizer_calls'],isolated_calls=273,formal=600)
 for seed in (41,42,43):
  case=suite+'_'+str(seed);p=R/case;layout={};exec((p/'layout.py').read_text(),layout)
  assert layout['ARMS']=='C' and layout['SEED']==seed and layout['SUITE']==suite
  st=layout['STAGE_EPOCHS'];assert st=={6:100,7:200,8:200,9:200}
  production=sum(((v+9)//10)*st[t] for t,v in layout['POP'].items());assert production==layout['TOTAL_UPDATES']
  ref=js(p/'S_REFERENCE.json');prefix='seed%d/'%seed
  for n,v in ref.items():
   if n=='stage6_checkpoint':
    assert v['sha256']==json.loads(e['light'][prefix+'STAGE_S_6.json']['text'])['sha256'];continue
   x=e['light'][prefix+n];assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256']==v['sha256'],(case,n)
  sh=json.loads(e['light'][prefix+'EPOCHS_S.json']['text']);assert len(sh)==400
  own=0;by={}
  for t in range(6,10):
   ss=[x for x in sh if x['stage']==t];assert len(ss)==100
   calls=sum(x['batches']*(1 if t==6 else 2) for x in ss);assert calls==((layout['POP'][t]+9)//10)*st[t]
   own+=json.loads(e['light'][prefix+'ADMISSION_S_%d.json'%t]['text'])['optimizer_calls'];by[t]=calls
  assert own==91
  for f in p.glob('*.py'):ast.parse(f.read_text(),feature_version=(3,8))
  train=(p/'train.py').read_text();runtime=(p/'runtime.py').read_text()
  assert 'for ep in range(STAGE_EPOCHS[stage])' in train and 'len(history)==700' in train
  assert "CALLS=={'production':TOTAL_UPDATES,'admission':91}" in train
  assert 'a.total_steps=STAGE_EPOCHS[ds.stage]*len(loader)' in runtime
  assert train.count('calls+=24')==1 and 'refill' not in train
  cases[case]=dict(stage_production_calls=by,production_calls=production,isolated_calls=91,new_formal=200,new_S_calls=0,source_S_reference=ref)
assert sum(x['production_calls'] for x in cases.values())==1012500
review=dict(status='PASS_STATIC_BUDGET_AND_ACCEPTED_S_BINDINGS_NOT_GPU_ADMISSION',time=datetime.datetime.now().astimezone().isoformat(),cases=cases,sources=out,local_build_note='Initial raw-byte parent binding check failed because local files use CRLF. Canonical transported UTF8 LF hashes exactly match all six remote frozen parent files; no remote scientific commands executed for that local failure.',limitations=['Own C GPU admission and actual scheduler checks pending dispatch','Final CPU wrappers static only','Original OBJECT engineering/interruption cost remains separate; no historical total-FLOPs equality'])
(R/'LOCAL_BUILD_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(R/'S_ACCEPTANCE.json').write_text(json.dumps(dict(status='ACCEPTED_S_READONLY_BOUND',sources=out,cases=cases),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=review['status'],cases=len(cases),new_C_production=1012500,new_C_isolated=546,new_S_training=0,new_formal=1200)))
