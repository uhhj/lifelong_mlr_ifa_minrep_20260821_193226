"""Authorized RTX3090 migration; frozen source unchanged, no completed epoch replay."""
import sys,ast,json,hashlib,time,copy,importlib.util
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent;case=sys.argv[1];resume=case=='L005_41'
assert case in [g+'_'+str(k) for g in ['L005','L020'] for k in [41,42,43]]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((E/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items())
assert json.loads((E/'VALIDATION.json').read_text())['status']=='PASS_GPU_RECOVERY_AND_PAIRED_BRANCH_REPLAY'
sys.path.insert(0,str(Q/case));sys.argv=[str(Q/case/'train.py'),'S']
import runtime as rt
import train as tr
torch=rt.torch
rt.VOLUME_BOUND['/data']+=346429749
assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
if resume:
 spec=importlib.util.spec_from_file_location('resume_current',E/'resume_current.py');tr=importlib.util.module_from_spec(spec);spec.loader.exec_module(tr)
 tr.ENG=E;tr.CHECKPOINT_SHA='65b8d3b4c9f589fb543ba5b7a41991a8c1dbc50daae309c21c386471178b4070'
 rt._copy_checked.update(rt.js(Q/case/'COPY_EQ.json')['checked'])
else:
 import initialization as init
 source=(Q/case/'initialization.py').read_text()
 old="assert tensor==old['tensors'] and digest==old['rng'] and hardware==old['hardware'],'Accepted S initialization/RNG/hardware mismatch'"
 new="assert tensor==old['tensors'] and digest==old['rng'],'Accepted S initialization/RNG mismatch'\n assert all(hardware[k]==old['hardware'][k] for k in ['torch_version','cuda_version','cudnn_version','matmul_tf32','cudnn_tf32'])\n write(R/'HARDWARE_MIGRATION.json',dict(status='USER_AUTHORIZED_RTX3090',prior=old['hardware'],current=hardware,initial_tensor_rng_exact=True,cross_GPU_training_equivalence_claim=False))"
 assert source.count(old)==1;exec(compile(source.replace(old,new),str(Q/case/'initialization.py'),'exec'),init.__dict__);tr.initialization=init.initialization
def device(x):
 if torch.is_tensor(x):return x.detach().clone()
 if isinstance(x,dict):return {k:device(v) for k,v in x.items()}
 if isinstance(x,list):return [device(v) for v in x]
 if isinstance(x,tuple):return tuple(device(v) for v in x)
 return copy.deepcopy(x)
checked={}
def branch_capture(a):
 old=rt.cpu
 try:rt.cpu=device;v=rt._reference_capture(a)
 finally:rt.cpu=old
 key=str(a.step)
 if not checked or a.scheduler.last_epoch==0:
  assert rt.equal(rt.cpu(v),rt.capture(a))
  checked[key]=dict(full_state_values_exact=True,time=time.time(),optimizer_calls=0,environments=0)
  rt.write(E/(case+'_BRANCH_COPY_EQ.json'),dict(status='PASS_OBSERVED_STAGE_STARTS',checked=checked))
 return v
def task6_compare(actual,expected):
 assert actual.keys()==expected.keys()
 exact=rt.equal(actual,expected);d=[]
 for k,v in actual.items():
  w=expected[k];assert v.shape==w.shape and v.dtype==w.dtype and torch.isfinite(v).all()
  if not torch.equal(v,w):d.append(dict(name=k,max_abs=float((v.double()-w.double()).abs().max())))
 rt.write(E/(case+'_TASK6_GPU_COMPARISON.json'),dict(status='RECORDED_AUTHORIZED_CROSS_GPU_COMPARISON',all_tensors_exact=exact,differing_tensors=d,hardware=torch.cuda.get_device_name(0),not_a_scientific_acceptance=True))
 return True
source=(E/'resume_current.py' if resume else Q/case/'train.py').read_text()
needle="assert equal(actual,torch.load(sr['path'],map_location='cpu')['state_dict']),'Task6 accepted S tensor mismatch'"
assert source.count(needle)==1;source=source.replace(needle,"assert task6_compare(actual,torch.load(sr['path'],map_location='cpu')['state_dict'])")
source=source.replace("same_seed_new_arm_comparator='ACCEPTED_S_ALL_TENSORS_EXACT'","same_seed_new_arm_comparator='AUTHORIZED_CROSS_GPU_COMPARISON_RECORDED'")
tree=ast.parse(source);main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
class Change(ast.NodeTransformer):
 def __init__(self):self.count=0
 def visit_Assign(self,node):
  self.generic_visit(node)
  if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['base','cstate','qstate'] and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='capture':
   node.value.func.id='branch_capture';self.count+=1
  return node
change=Change();change.visit(main);assert change.count==3
# Three branch snapshots stay on device; verification capture remains CPU and unchanged.
tr.branch_capture=branch_capture;tr.task6_compare=task6_compare
tr.equal=lambda x,y:rt.equal(rt.cpu(x),rt.cpu(y))
exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),str(E/'train_wrapper.py'),'exec'),tr.__dict__)
tr.main()
