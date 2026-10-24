import os,sys,json,hashlib,ast,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())
results=[]
for e in SPEC['seeds']:
 O=Path(e['output']);S=Path(e['source']);assert sha(O/'CONTRACT.json')==e['contract_sha256']
 c=js(O/'CONTRACT.json');assert all(sha(S/k)==h for k,h in c['source_sha256'].items())
 assert (O/'pipeline.exit').read_text().strip()==(O/'supervisor.exit').read_text().strip()=='0'
 for k,h in e['metadata'].items():assert sha(O/k)==h
 for t,h in e['checkpoints'].items():assert sha(O/'ra_coreset/models'/('mlrifa_'+t+'_model_ep99.pth'))==h
 side=O/'effects/memory/task8.pth';m=js(side.with_suffix('.json'));assert sha(side)==m['sha256']
 labels=torch.load(side,map_location='cpu');selected=js(O/'ra_coreset/storage/task8.json')['selected'];assert len(selected)==len(labels)==452
 row=labels[selected[0]['destination']]
 results.append({'seed':e['seed'],'contract_sha256':e['contract_sha256'],'source_files':len(c['source_sha256']),'label_keys':list(row),'next_action_shape':list(row['next_action'].shape),'valid':sum(r['bc_valid'] for r in labels.values()),'selected_count':len(selected)})
S=Path(SPEC['seeds'][0]['source'])
sources={n:(S/n).read_text() for n in ['libero/lifelong/models/policy_head.py','libero/lifelong/models/brief.py','libero/lifelong/models/modules/transformer_modules.py']}
print(json.dumps({'kind':'preflight','status':'PASS_ZERO_FORWARD_PREFLIGHT','results':results,'source':sources,'forwards':0,'updates':0,'environments':0,'cuda':torch.cuda.is_initialized(),'time':time.time()}))
