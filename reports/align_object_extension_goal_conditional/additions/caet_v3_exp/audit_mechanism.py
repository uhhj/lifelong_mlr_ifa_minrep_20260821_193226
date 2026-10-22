"""Independent ledger recomputation; no training or environments."""
import json,random,math
from pathlib import Path
from collections import Counter
import torch
from coreset_exp.common import O,SEED,sha
from .variant import VARIANT
from .eftr_optimizer import choose
def js(p):return json.loads(p.read_text())
def audit():
 root=O/'ra_coreset';counts=Counter();headsteps=0;cursor=0
 for t in range(6,10):
  manifest=js(root/'dataset_manifests'/('task%d.json'%t));rows=manifest['rows'];n=len(rows);positive=manifest['positive'];negative=manifest['negative']
  assert n==js(O/'CONTRACT.json')['dataset_sizes'][str(t)]
  paths=sorted([str(p) for p in (root/'buffer').glob('*.pt') if int(p.name.split('__')[0][1:])<t])+sorted(str(p) for p in (O/'features'/str(t)).glob('*.pt')) if t>6 else sorted(str(p) for p in (O/'features'/str(t)).glob('*.pt'))
  random.Random(SEED).shuffle(paths);assert paths==[r['path'] for r in rows]
  labels={str(O/'features'/str(t)/name):v for name,v in torch.load(O/'effects/current'/('task%d'%t)/'labels.pth',map_location='cpu').items()}
  for j in range(6,t):
   p=O/'effects/memory'/('task%d.pth'%j);assert sha(p)==js(p.with_suffix('.json'))['sha256']
   labels.update({str(root/'buffer'/name):v for name,v in torch.load(p,map_location='cpu').items()})
  for r in rows:
   label=labels[r['path']];assert sha(r['path'])==r['sha256']==label['sample_sha256'];assert r['bc_valid']==label['bc_valid'] and r['effect_valid']==label['valid']
  oldtasks={j:sorted([r['index'] for r in rows if r['task']==j],key=lambda i:rows[i]['sample']) for j in range(6,t)}
  old=[v[k] for k in range(max([len(v) for v in oldtasks.values()] or [0])) for j,v in sorted(oldtasks.items()) if k<len(v)];cursor=0
  if VARIANT=='EFTR':
   p=O/'teachers'/('task%d.pth'%t);meta=js(p.with_suffix('.json'));assert sha(p)==meta['sha256'] and sha(root/'models'/('mlrifa_%d_model_ep99.pth'%t))==meta['checkpoint_sha256']
   teacher=torch.load(p,map_location='cpu');storage=js(root/'storage'/('task%d.json'%t));assert list(teacher)==[x['destination'] for x in storage['selected']]
   for item in storage['selected']:
    row=teacher[item['destination']];assert row['sample_sha256']==item['sha256'];assert row['mode']==int(row['logits'].argmax());assert row['means'].shape==row['std'].shape==(5,7);assert (row['std']>0).all()
  for epoch in range(100):
   order=torch.randperm(n,generator=torch.Generator().manual_seed(epoch)).tolist();stats=js(root/'auxiliary'/('task%d_epoch%02d.json'%(t,epoch)));assert len(stats)==math.ceil(n/10)
   for b,s in enumerate(stats):
    ix=order[b*10:b*10+10];assert s['batch_ids']==ix
    valid=sum(rows[i]['bc_valid'] for i in ix) if VARIANT!='EFTR' else len(ix)
    assert s['bc_valid']==valid and s['bc_invalid']==len(ix)-valid and s['committed_updates']==1
    counts['bc_exposures']+=valid;counts['invalid_bc_exposures']+=len(ix)-valid;counts['main_updates']+=1
    if VARIANT=='TAC':
     em=sum(rows[i]['effect_valid'] for i in ix);anchors=[i for i in ix if positive.get(str(i)) and negative.get(str(i))]
     expected=[]
     for i in anchors:
      pos=positive[str(i)];neg=negative[str(i)];expected.append([pos[s['step']%len(pos)]]+[neg[(s['step']+k)%len(neg)] for k in range(min(4,len(neg)))])
     assert s['effect_samples']==em and s['anchor_ids']==anchors and s['partner_ids']==expected
     assert s['extra_partner_forwards']==sum(len(z) for z in expected)
     headsteps+=int(bool(em or anchors));assert s['head_steps']==headsteps and s['head_updates']==int(bool(em or anchors))
     counts['head_updates']+=s['head_updates'];counts['effect_exposures']+=em;counts['cross_anchors']+=len(anchors);counts['partner_forwards']+=s['extra_partner_forwards']
    if VARIANT=='EFTR' and t>6:
     ids=[old[(cursor+k)%len(old)] for k in range(min(32,len(old)))];cursor=(cursor+len(ids))%len(old)
     assert s['old_check_ids']==ids and s['cursor']==cursor and s['optimizer_trials']==3
     chosen,unmet=choose(s['scores']);assert s['selected']==chosen and s['budget_unmet']==unmet
     counts['budget_unmet']+=unmet;counts['selected_'+str(chosen)]+=1;counts['old_check_exposures']+=len(ids)
    counts['optimizer_trials']+=s['optimizer_trials']
  if VARIANT=='TAC':
   cp=torch.load(root/'aux_heads'/('task%d.pth'%t),map_location='cpu');assert cp['steps']==headsteps and len(cp['model'])==8
 assert counts['main_updates']==js(O/'CONTRACT.json')['optimizer_steps']
 return dict(status='PASS',variant=VARIANT,counts=dict(counts),boundary_masks_and_native_order_verified=True,actual_head_or_candidate_updates_separately_counted=True)
