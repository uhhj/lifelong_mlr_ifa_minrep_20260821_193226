"""Additional zero-training OOD and frozen predictor audits after policy forwards."""
result['semantic_source_sha256']=REQUEST['semantic_source_sha256'];result['semantic_rule']='OPEN negative / HOLD exact zero / CLOSE positive; stateful PandaGripper sign increment'
result['ood']={};result['predictors']=[]
for task in (8,9):
 g=REQUEST['stages'][str(task)];supported={g['current_nodes'][i]['sample'] for i,j in g['edges']};rows=[r for r in allrows[task] if not r['holdout'] and r['sample'] in supported];a=torch.stack([r['next_action'] for r in rows]);base=local_query(rows,a,task);stress={}
 for label in ('all_plus10sigma','all_minus10sigma')+tuple('dim%d_%s'%(k,sg) for k in range(6) for sg in ('minus1','plus1')):
  altered=a.clone()
  if label.startswith('all_'):
   for i,r in enumerate(rows):altered[i,:6]+=(10 if label=='all_plus10sigma' else -10)*estimator(task,r['demo'])[1][:6]
  else:
   dim=int(label[3]);altered[:,dim]=-1. if label.endswith('minus1') else 1.
  vals=local_query(rows,altered,task);stress[label]=dict(n=len(rows),accepted=sum(v['accepted'] for v in vals),rows=[dict(identity=ident(r),local=v) for r,v in zip(rows,vals)])
 equivalence={}
 for magnitude in (1e-6,.2,2.):
  altered=a.clone();altered[:,6]=torch.sign(a[:,6])*magnitude;vals=local_query(rows,altered,task);assert vals==base
  equivalence[str(magnitude)]=dict(queries=len(rows),support_and_weight_exact=True)
 result['ood'][str(task)]=dict(stress=stress,gripper_same_class=equivalence)
 print('Semantic and OOD checks task %s complete'%task,file=sys.stderr,flush=True)
def phase(r):
 parts=r['group'].split('/');return ('STATIC' if parts[1]=='STATIC' else 'TRANSLATION' if parts[1].startswith('T') else 'ROTATION')+'/'+parts[-1]
class FrozenEffect(torch.nn.Module):
 def __init__(self):
  super().__init__();self.body=torch.nn.Sequential(torch.nn.Linear(1543,256),torch.nn.ReLU(),torch.nn.Linear(256,128),torch.nn.ReLU());self.output=torch.nn.Linear(128,7)
 def forward(self,x,a):return self.output(self.body(torch.cat((x[:,:,1:,:].mean(1).reshape(len(x),1536),a),-1)))
for task in (8,9):
 root=Path('/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed41_20261021/effects/models')/('task%d'%task);cp=root/'final.pth';h=sha(cp);saved=torch.load(cp,map_location='cpu');admission=js(root/'ACCEPTANCE.json');assert admission['frozen_predictor_sha256']==h
 m=FrozenEffect();m.load_state_dict(saved['model'],strict=True);m.eval().requires_grad_(False)
 # Current windows plus only retained old RA members, exactly the effect fitting population.
 rows=allrows[task]+[r for t in range(6,task) for r in oldrows[t]];tr=[r for r in rows if not r['holdout']];va=[r for r in rows if r['holdout']]
 X=torch.stack([torch.load(P/'features'/str(r['task'])/r['sample'],map_location='cpu')['feature'].squeeze(0) for r in va]);A=torch.stack([r['next_action'] for r in va]);Y=torch.tensor(np.stack([r['effect'] for r in va]),dtype=torch.float32);trainY=torch.tensor(np.stack([r['effect'] for r in tr]),dtype=torch.float32)
 target=(Y-saved['center'])/saved['scale'];train_target=(trainY-saved['center'])/saved['scale']
 with torch.no_grad():
  pred=m(X,A);global_perm=m(X,A.roll(1,0));equiv={}
  for magnitude in (1e-6,.2,2.):
   aa=A.clone();aa[:,6]=aa[:,6].sign()*magnitude;equiv[str(magnitude)]=m(X,aa)
 # Phase and task-conditioned arm-only cyclic permutation does not change gripper class.
 perm=A.clone();changed=torch.zeros(len(va),dtype=torch.bool)
 for key in sorted({(r['task'],phase(r),int(torch.sign(r['next_action'][6]))) for r in va}):
  ids=[i for i,r in enumerate(va) if (r['task'],phase(r),int(torch.sign(r['next_action'][6])))==key]
  perm[ids,:6]=A[ids,:6].roll(1,0);changed[ids]=(perm[ids,:6]!=A[ids,:6]).any(-1)
 with torch.no_grad():permpred=m(X,perm)
 aa=A.clone().requires_grad_(True);gradient=torch.autograd.grad(m(X,aa).square().mean(),aa)[0];assert torch.isfinite(gradient).all()
 phase_rows={}
 for ph in sorted({phase(r) for r in va}):
  ids=[i for i,r in enumerate(va) if phase(r)==ph];trainids=[i for i,r in enumerate(tr) if phase(r)==ph];assert trainids
  mean=train_target[trainids].mean(0);values=dict(expert_mae=float((pred[ids]-target[ids]).abs().mean()),global_mean_mae=float(target[ids].abs().mean()),phase_mean_mae=float((target[ids]-mean).abs().mean()),arm_permutation_mae=float((permpred[ids]-target[ids]).abs().mean()))
  demos=sorted({(va[i]['task'],va[i]['demo']) for i in ids});sens=float(gradient[ids,:6].abs().sum())
  eq={k:dict(mae=float((v[ids]-target[ids]).abs().mean()),prediction_drift=float((v[ids]-pred[ids]).abs().mean())) for k,v in equiv.items()}
  reasons=[]
  if len(ids)<12:reasons.append('FEWER_THAN_12_WINDOWS')
  if len(demos)<2:reasons.append('FEWER_THAN_2_TASK_DEMOS')
  if int(changed[ids].sum())<12:reasons.append('INSUFFICIENT_ARM_PERMUTATION_CHANGE')
  for k in ('global_mean_mae','phase_mean_mae','arm_permutation_mae'):
   if values['expert_mae']>=values[k]:reasons.append('NOT_BETTER_THAN_'+k)
  if sens<=0:reasons.append('ZERO_ARM_GRADIENT')
  if any(v['mae']>=values['phase_mean_mae'] for v in eq.values()):reasons.append('SEMANTIC_GRIPPER_VARIANTS_NOT_BETTER_THAN_PHASE_MEAN')
  current=[i for i in ids if va[i]['task']==task]
  phase_rows[ph]=dict(windows=len(ids),task_demos=demos,changed_arm_permutations=int(changed[ids].sum()),arm_gradient_abs_sum=sens,metrics=values,semantic_variants=eq,accepted=not reasons,reasons=reasons,current_only=dict(windows=len(current),demos=sorted({va[i]['demo'] for i in current}),expert_mae=float((pred[current]-target[current]).abs().mean()) if current else None))
 global_mae=float((pred-target).abs().mean());global_baseline=float(target.abs().mean());oldperm=float((global_perm-target).abs().mean())
 assert abs(global_mae-admission['heldout_mae'])<1e-5,(global_mae,admission['heldout_mae'])
 assert sha(cp)==h
 result['predictors'].append(dict(task=task,model='frozen_original_CAET_v2_mean8_not_new_RSET',sha256=h,unchanged=True,training_windows=len(tr),holdout_windows=len(va),global_recomputed=dict(mae=global_mae,mean_mae=global_baseline,global_permutation_mae_order_not_native=oldperm),original_admission=admission,phases=phase_rows,holdout_identities=[ident(r) for r in va],all_parameters_frozen=True,optimizer_calls=0))
 del m,saved,X,A,Y
 print('Frozen predictor strata task %s complete'%task,file=sys.stderr,flush=True)
result['predictor_gate']='FROZEN_V2_STRATIFIED_AUDIT_ONLY_NEW_RSET_UNVALIDATED'
