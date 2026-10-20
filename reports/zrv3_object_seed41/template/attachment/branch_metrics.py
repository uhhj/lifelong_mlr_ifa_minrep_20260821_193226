"""Exact record-only per-parent comparison. The multi-seed controller owns the gate."""
from pathlib import Path
from fractions import Fraction
import hashlib,json
from zr3_exp.params import SEED,PARM,ARM
from compare_metrics import metrics,_sha_field
def load_records(folder,is_new):
 folder=Path(folder);expected={'trials_stage%d_task%d.jsonl'%(s,t) for s in range(6,10) for t in range(6,s+1)}
 assert {p.name for p in folder.glob('trials*.jsonl')}==expected
 rates={};records={};hashes={};stages={};modals={}
 for s in range(6,10):
  for t in range(6,s+1):
   f=folder/('trials_stage%d_task%d.jsonl'%(s,t));raw=f.read_bytes();hashes[f.name]=hashlib.sha256(raw).hexdigest()
   rows=[json.loads(l) for l in raw.decode('utf-8-sig').splitlines() if l.strip()]
   assert len(rows)==20 and {r['trial'] for r in rows}==set(range(20))
   assert {r['initial_state_index'] for r in rows}==set(range(20))
   for r in rows:
    assert (r['seed'],r['stage'],r['task'],r['arm'])==(SEED,s,t,ARM if is_new else PARM)
    assert type(r['success']) is bool and r['trial']==r['initial_state_index']
    for k in ['checkpoint_sha256','initial_state_sha256','modal_sha256']:_sha_field(r[k],k)
    if is_new:
     assert r['parent_epoch']==99 and r['calibration_epoch']==9
     cp=_sha_field(r['parent_checkpoint_sha256'],'parent_checkpoint_sha256')
    else:assert r['epoch']==99;cp=r['checkpoint_sha256']
    assert s not in stages or stages[s]==cp;stages[s]=cp
    assert s not in modals or modals[s]==r['modal_sha256'];modals[s]=r['modal_sha256']
    records[s,t,r['trial']]=r
   rates[s,t]=Fraction(sum(r['success'] for r in rows),20)
 return rates,records,hashes
def compare(baseline,new):
 br,b,bh=load_records(baseline,False);nr,n,nh=load_records(new,True);bm=metrics(br);nm=metrics(nr)
 for key in b:
  for f in ['initial_state_index','initial_state_sha256','modal_sha256']:assert b[key][f]==n[key][f]
  assert b[key]['checkpoint_sha256']==n[key]['parent_checkpoint_sha256']
 pack=lambda m:{k:dict(fraction=str(v),percent=float(v*100)) for k,v in m.items()}
 return dict(status='PASS_RECORD_COMPARISON',baseline=pack(bm),new=pack(nm),delta_pp=pack({k:nm[k]-bm[k] for k in bm}),metric_gate=nm['AUC']>bm['AUC'] and nm['FWT']>=bm['FWT'] and nm['OldFinal']>=bm['OldFinal'],metric_gate_scope='Single diagnostic seed41: AUC improves, FWT and OldFinal do not decrease; engineering acceptance also required',baseline_files_sha256=bh,new_files_sha256=nh,matrices={name:[[float(rates[s,t]*100) if t<=s else None for t in range(6,10)] for s in range(6,10)] for name,rates in [('parent',br),('new',nr)]})
