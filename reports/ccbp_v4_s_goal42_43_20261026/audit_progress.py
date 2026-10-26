"""Recompute both seeds independently, preserving top-level dispatch/health evidence."""
import json,sys,subprocess,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'));v=js(D/'EVIDENCE.json')
for n,x in v['light'].items():assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256']
L={n:json.loads(x['text']) if n.endswith('.json') else x['text'] for n,x in v['light'].items()}
failure=L.get('FAILURE.json')
if not failure and L.get('controller.exit')!='0' and not v.get('processes'):failure=dict(reason='No success exit and no own process')
audits={};evaluations={}
for seed in (42,43):
 q=D/('seed%d'%seed);q.mkdir(exist_ok=True);prefix='seed%d/'%seed
 child=dict(v,contract_sha256=v['children'][str(seed)],light={n[len(prefix):]:x for n,x in v['light'].items() if n.startswith(prefix)})
 if 'controller.exit' in v['light']:child['light']['controller.exit']=v['light']['controller.exit']
 child['light']['STATUS.json']=v['light']['STATUS.json']
 (q/'EVIDENCE.json').write_text(json.dumps(child),encoding='utf8')
 subprocess.run([sys.executable,'-X','utf8',str(R/'audit_seed_progress.py'),str(q),str(seed)],check=True,capture_output=True,text=True,encoding='utf8')
 audits[str(seed)]=js(q/'LOCAL_AUDIT.json')
 if 'OUTCOMES.json' in child['light']:
  subprocess.run([sys.executable,'-X','utf8',str(R/'audit_seed_evaluation.py'),str(q),str(seed)],check=True,capture_output=True,text=True,encoding='utf8');evaluations[str(seed)]=js(q/'EVALUATION_PROGRESS_AUDIT.json')
out=dict(status='FAILURE_REQUIRES_INSPECTION' if failure else 'PASS_PROGRESS_NOT_FINAL',observation=v['observed'],contract=v['contract_sha256'],phase=L.get('STATUS.json'),failure=failure,seeds=audits,evaluations=evaluations,total_epochs=sum(a['counts']['S']['epochs'] for a in audits.values()),total_main_updates=sum(a['counts']['S']['steps'] for a in audits.values()),training_collection_episodes=sum(a['training_collection_episodes'] for a in audits.values()),formal_episodes=sum(a['formal_episodes'] for a in audits.values()),isolated_optimizer_updates=sum(a['isolated_optimizer_calls'] for a in audits.values()),source_files=v['source_files'],evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest())
assert all(a['failure'] is None for a in audits.values()) or failure
(D/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:x for k,x in out.items() if k not in ['seeds','evaluations']},indent=2))
