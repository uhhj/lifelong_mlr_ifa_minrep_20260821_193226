"""Build external recovery wrappers only; never execute frozen experiment entrypoints."""
from pathlib import Path
import ast,json,hashlib,difflib
R=Path(__file__).resolve().parent;E=R/'engineering_resume_3090_20261102_v1';E.mkdir(exist_ok=True)
def write(n,s):
 ast.parse(s);(E/n).write_text(s,encoding='utf8')
src=(R/'L005_41/train.py').read_text(encoding='utf8')
s=src
def replace(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
replace("assert not (dest/'recovery.pth').exists(),'No blind resume'", "assert sha(dest/'recovery.pth')==CHECKPOINT_SHA, 'Interrupted checkpoint changed'")
replace("a=make(arm);initialization(a,arm);history=[];gradients=[];gates=[];physical=0", "a=make(arm);cp=torch.load(dest/'recovery.pth',map_location='cpu');assert cp['history']==js(R/'EPOCHS_S.json') and len(cp['history'])==260\n        history=cp['history'];gradients=js(R/'GRADIENTS_S.json');gates=[json.loads(line) for line in (R/'GATES.jsonl').read_text().splitlines()];assert len(gates)==4160\n        physical=cp['physical_optimizer_calls'];CALLS.update(production=physical,admission=62)")
replace('for stage in range(6,10):','for stage in range(8,10):')
start=s.index("            set_count_mode('admission');admission_before=")
end=s.index("            set_count_mode('production')",start)
ad=s[start:end]
s=s[:start]+"            if stage==8:\n                assert binding==cp['memory_bindings'] and cp['probe_sha256']==sha(R/'PROBES_8.json')\n                assert json.loads(json.dumps(pr['metadata']))==js(R/'PROBES_8.json')\n                restore(a,cp['state']);assert equal(cp['state'],capture(a))\n                write(ENG/'PRODUCTION_RESTORE.json',dict(status='PASS_FULL_STATE_RESTORED',time=time.time(),step=a.step,checkpoint_sha256=CHECKPOINT_SHA,production_updates_added=0))\n            else:\n"+''.join('    '+line+'\n' for line in ad.splitlines())+s[end:]
replace('for ep in range(100):','for ep in range(60 if stage==8 else 0,100):')
# Keep main body identical apart from resume boundaries; branch-copy optimization is installed by shared wrapper.
write('resume_current.py',s.replace("if __name__=='__main__':main()",''))
(E/'resume_current.diff').write_text(''.join(difflib.unified_diff(src.splitlines(True),s.splitlines(True),fromfile='frozen/train.py',tofile='resume_current.py')),encoding='utf8')
write('train_wrapper.py',r'''"""Authorized RTX3090 migration; frozen source unchanged, no completed epoch replay."""
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
exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),str(E/'train_wrapper.py'),'exec'),tr.__dict__)
tr.main()
''')
write('controller.py',r'''"""Unique post-reboot owner; resumes unfinished epoch then original suite order."""
import os,sys,json,time,hashlib,fcntl,subprocess,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent;PY=sys.executable
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def verify():
 for c in [Q/'CONTRACT.json',Q/'engineering_device_snapshot_20261101_v1/CONTRACT.json',E/'CONTRACT.json']:
  assert all(sha(Path(p))==h for p,h in js(c)['files'].items())
def main():
 lock=Path('/data/mlr_ifa/'+Q.name+'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (E/'STARTED.json').exists();verify();assert js(E/'VALIDATION.json')['status']=='PASS_GPU_RECOVERY_AND_PAIRED_BRANCH_REPLAY'
 put(E/'STARTED.json',dict(pid=os.getpid(),time=time.time()))
 for group in ['L005','L020']:
  if group=='L020':assert (Q/'L005_EVALUATION_COMPLETE.json').exists()
  for phase in ['train','evaluate']:
   for seed in [41,42,43]:
    case=group+'_'+str(seed);r=Q/case;name='train_S' if phase=='train' else 'evaluate';verify()
    assert not (r/(name+'.exit')).exists()
    argv=[PY,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(E/'train_wrapper.py'),case] if phase=='train' else [PY,'-B',str(r/'evaluate.py')]
    logfile=E/(case+'_resume_train.log') if phase=='train' and case=='L005_41' else r/(name+'.log')
    with logfile.open('x') as f:
     p=subprocess.Popen(argv,cwd=r,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
     status=dict(phase=phase,suite=group,seed=seed,pid=p.pid,time=time.time(),controller=os.getpid(),engineering_contract=sha(E/'CONTRACT.json'),hardware='RTX3090',completion_source=str(E/'controller.exit'))
     put(E/'STATUS.json',status);put(Q/'STATUS.json',status);rc=p.wait()
    (r/(name+'.exit')).write_text(str(rc));assert rc==0,(case,phase,rc)
  assert all(len(js(Q/(group+'_'+str(k))/'OUTCOMES.json'))==200 for k in [41,42,43])
  put(Q/(group+'_EVALUATION_COMPLETE.json'),dict(status='PASS_SUITE_EXECUTION_PENDING_FINAL_CPU',episodes=600,time=time.time()))
 assert not (Q/'controller.exit').exists() and not (Q/'engineering_device_snapshot_20261101_v1/controller.exit').exists()
 put(Q/'STATUS.json',dict(phase='complete',time=time.time(),completion_source=str(E/'controller.exit')));(E/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  put(E/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));(E/'controller.exit').write_text('1');raise
''')
print(str(E))
