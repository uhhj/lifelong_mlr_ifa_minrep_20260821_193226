"""Read-only CPU diagnosis. No optimizer, forward pass, environment or recovery."""
import os,json,hashlib,time,sys,ast
from pathlib import Path
import torch
Q=Path('/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019')
W=Q/'seed41';O=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019')
P=Path('/root/mlr_ifa_new_experiments/Seeds40_41_three_arm_20261016/seed41/rc')
PS=Path('/data/mlr_ifa/experiment_workspaces/Seeds40_41_three_arm_20261016/seed41/rc/source')
E=Path(sys.argv[1]);E.mkdir(parents=True,exist_ok=True)
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def flat(x,p=''):
 if isinstance(x,dict):
  return {k:v for n,y in x.items() for k,v in flat(y,p+'.'+str(n)).items()}
 return {p:x}
def main():
 torch.set_num_threads(2)
 cp=O/'ra_coreset/models/mlrifa_6_model_ep99.pth';pc=P/'coreset/models/mlrifa_6_model_ep99.pth'
 inputs=[cp,pc,O/'ra_coreset/recovery/latest.pth',O/'CONTRACT.json',P/'CONTRACT.json',O/'ra_coreset/INITIAL_STATE.json',P/'coreset/INITIAL_STATE.json',O/'ra_coreset/epochs.json',P/'coreset/epochs.json']
 before={str(p):sha(p) for p in inputs}
 a=torch.load(cp,map_location='cpu');b=torch.load(pc,map_location='cpu');aa=a['state_dict'];bb=b['state_dict'];assert set(aa)==set(bb)
 rows=[]
 for k in sorted(aa):
  x=aa[k];y=bb[k];assert x.shape==y.shape and x.dtype==y.dtype
  d=(x.double()-y.double()).abs()
  rows.append(dict(key=k,equal=torch.equal(x,y),count=x.numel(),different=int((x!=y).sum()),max_abs=float(d.max()),mean_abs=float(d.mean()),rmse=float(d.square().mean().sqrt()),parent_norm=float(y.double().norm()),difference_norm=float(d.norm())))
 fa=flat(dict(a['cfg']));fb=flat(dict(b['cfg']));cfgdiff={k:dict(current=fa.get(k),parent=fb.get(k)) for k in set(fa)|set(fb) if fa.get(k)!=fb.get(k)}
 ea=js(O/'ra_coreset/epochs.json');eb=[r for r in js(P/'coreset/epochs.json') if r['task']==6]
 epochs=[dict(epoch=x['epoch'],current_loss=x['loss'],parent_loss=y['loss'],loss_delta=x['loss']-y['loss'],batches_equal=x['batches']==y['batches'],lr_first_equal=x['lr_first']==y['lr_first'],lr_after_equal=x['lr_after']==y['lr_after']) for x,y in zip(ea,eb)]
 recovery=torch.load(O/'ra_coreset/recovery/latest.pth',map_location='cpu')
 current=js(O/'ra_coreset/INITIAL_STATE.json');parent=js(P/'coreset/INITIAL_STATE.json')
 # AST confirms update/build semantics independently of paths and checkpoint auditing.
 def nodes(path):
  tree=ast.parse(path.read_text());return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 an=nodes(W/'source/coreset_exp/train.py');bn=nodes(PS/'coreset_exp/train.py')
 semantic={k:an[k]==bn[k] for k in ['references','PaperMLR','build','capture','restore']}
 files=['libero/lifelong/algos/mlr.py','libero/lifelong/datasets.py','libero/lifelong/utils.py','libero/lifelong/models/brief.py']
 source_same={f:sha(W/'source'/f)==sha(PS/f) for f in files}
 from coreset_exp.common import verify
 verify()
 after={str(p):sha(p) for p in inputs};assert before==after
 out=dict(status='BLOCKED_TASK6_ACTUAL_TENSOR_PARITY',time=time.time(),readonly=True,optimizer_calls=0,environment_steps=0,frozen_contract_and_features='PASS',input_hashes_unchanged=True,input_hashes=before,
   initial_state_equal=current==parent,initial_rng_equal=current['rng']==parent['rng'],initial_tensors_equal=current['tensors']==parent['tensors'],tensor_rows=rows,different_tensor_count=sum(not r['equal'] for r in rows),total_tensor_count=len(rows),max_abs=max(r['max_abs'] for r in rows),config_differences=cfgdiff,update_AST_equal=semantic,critical_source_bytes_equal=source_same,epochs=epochs,
   recovery=dict(task=recovery['task'],next_epoch=recovery['next_epoch'],stage_complete=recovery['stage_complete'],step=recovery['state']['step'],buffer_entries=len(recovery['buffer_hashes'])),
   formal_epoch_rows=len(ea),formal_optimizer_steps=ea[-1]['steps'],committed_recovery_steps=recovery['state']['step'],evaluation_records=0,storage_manifests=len(list((O/'ra_coreset/storage').glob('task*.json'))),
   explanation='Task6 failed before selection; RA selection has not yet intervened. Cross-device/numerical execution is a possible contributor, not established cause. Exact tensor parity remains a frozen requirement. No automatic bypass or performance fallback is permitted.')
 dest=E/'TASK6_READONLY_DIAGNOSIS.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:out[k] for k in ['status','initial_state_equal','different_tensor_count','total_tensor_count','max_abs','update_AST_equal','critical_source_bytes_equal','recovery','formal_optimizer_steps','storage_manifests']}))
if __name__=='__main__':main()
