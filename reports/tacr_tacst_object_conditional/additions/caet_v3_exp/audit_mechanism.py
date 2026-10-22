"""Independent full native order, masks, fixed teachers and optimizer accounting."""
import json,random,math
from pathlib import Path
from collections import Counter
import torch
from coreset_exp.common import O,SEED,sha
from .variant import VARIANT
def js(p):return json.loads(p.read_text())
def audit():
 root=O/'ra_coreset';counts=Counter();step=0;teacher_proofs={}
 for t in range(6,10):
  md=js(root/'dataset_manifests'/('task%d.json'%t));rows=md['rows'];n=len(rows);assert n==[1562,2261,3081,3620][t-6]
  paths=sorted(str(p) for p in (root/'buffer').glob('*.pt') if int(p.name.split('__')[0][1:])<t)+sorted(str(p) for p in (O/'features'/str(t)).glob('*.pt'))
  random.Random(SEED).shuffle(paths);assert paths==[r['path'] for r in rows]
  labels={str(O/'features'/str(t)/k):v for k,v in torch.load(O/'effects/current'/('task%d'%t)/'labels.pth',map_location='cpu').items()}
  for j in range(6,t):
   p=O/'effects/memory'/('task%d.pth'%j);assert sha(p)==js(p.with_suffix('.json'))['sha256']
   labels.update({str(root/'buffer'/k):v for k,v in torch.load(p,map_location='cpu').items()})
  for r in rows:
   v=labels[r['path']];assert sha(r['path'])==r['sha256']==v['sample_sha256'];assert (r['bc_valid'],r['effect_valid'])==(v['bc_valid'],v['valid'])
  pos=md['positive'];neg=md['negative'];cursor=0;teacher={}
  if VARIANT=='TACRST' and t>6:
   p=O/'teacher_keys'/('stage%d.pth'%t);m=js(p.with_suffix('.json'));assert m['sha256']==sha(p) and m['dimension']==128 and m['teacher_optimizer_calls']==0 and m['teacher_frozen']
   teacher=dict(main=sha(root/'models'/('mlrifa_%d_model_ep99.pth'%(t-1))),student=sha(root/'aux_heads'/('task%d.pth'%(t-1))),keys=sha(p))
   assert m['identity']['teacher_main_sha256']==teacher['main'] and m['identity']['teacher_aux_sha256']==teacher['student'] and m['identity']['teacher_stage']==t-1
   assert m['identity']['support_sha256']==sha(O/'effects/current'/('task%d'%t)/'SUPPORT.json')
   keys=torch.load(p,map_location='cpu');ids=[i for i,r in enumerate(rows) if r['task']<t and r['bc_valid']]
   assert list(keys)==ids==[r['index'] for r in m['rows']] and len(ids)==m['count']
   for r in m['rows']:assert (r['sample'],r['sample_sha256'])==(rows[r['index']]['sample'],rows[r['index']]['sha256'])
   assert all(k.shape==(128,) and not k.requires_grad and torch.isfinite(k).all() and abs(float(k.norm())-1)<1e-5 for k in keys.values())
   teacher_proofs[str(t)]=dict(teacher,old_member_keys=len(keys),single_previous_stage=True)
  for ep in range(100):
   order=torch.randperm(n,generator=torch.Generator().manual_seed(ep)).tolist();stats=js(root/'auxiliary'/('task%d_epoch%02d.json'%(t,ep)));assert len(stats)==math.ceil(n/10)
   for b,s in enumerate(stats):
    ix=order[b*10:b*10+10];valid=sum(rows[i]['bc_valid'] for i in ix);em=sum(rows[i]['effect_valid'] for i in ix)
    assert s['batch_ids']==ix and s['step']==step and s['variant']==VARIANT and s['bc_valid']==valid and s['bc_invalid']==len(ix)-valid and s['effect_samples']==em
    assert s['committed_updates']==s['optimizer_trials']==s['representation_optimizer_calls']==s['head_updates']==1 and s['head_steps']==ep*len(stats)+b+1
    assert s['teacher_hash']==teacher and s['cursor_before']==cursor and 0<s['gate_mean']<=1 and 0<=s['residual_ratio_max']<=.200001
    anchors=[i for i in ix if pos.get(str(i)) and neg.get(str(i))] if VARIANT=='TACRST' and t>6 else [];partners=[]
    for i in anchors:
     ps=pos[str(i)];ns=neg[str(i)];selected=[ps[cursor%len(ps)]]+[ns[(cursor+k)%len(ns)] for k in range(min(4,len(ns)))];cursor+=1;partners.append(selected)
     assert rows[i]['task']==t and all(rows[j]['task']<t and rows[j]['effect_valid'] for j in selected)
     assert labels[rows[i]['path']]['grip_command_sign']==labels[rows[selected[0]]['path']]['grip_command_sign']
    assert s['anchor_ids']==anchors and s['partner_ids']==partners and s['cursor']==cursor and s['cross_anchors']==len(anchors) and s['old_key_reads']==sum(map(len,partners))
    if not anchors:assert s['st_loss']==0
    for name in ['bc','ifa','effect_loss','st_loss','gate_mean']:assert math.isfinite(s[name])
    counts.update(main_updates=1,representation_updates=1,bc_exposures=valid,invalid_bc_exposures=len(ix)-valid,effect_exposures=em,st_anchors=len(anchors),old_key_reads=sum(map(len,partners)))
    step+=1
  cp=torch.load(root/'aux_heads'/('task%d.pth'%t),map_location='cpu');assert len(cp['model'])==20 and cp['steps']==100*math.ceil(n/10)
  assert cp['scheduler']['last_epoch']==100*math.ceil(n/10)
 assert step==counts['main_updates']==counts['representation_updates']==105500
 return dict(status='PASS',variant=VARIANT,counts=dict(counts),teacher_proofs=teacher_proofs,native_order_masks_partner_rotation=True,disjoint_optimizer_calls_separate=True)
