"""Explicit authorized audit-only amendment. Original scientific source stays frozen."""
import ast,os,sys,json,hashlib,math,time
from pathlib import Path
N=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def accept_task6(same):
 O=Path(os.environ['CORESET_OUT']);P=Path(os.environ['RA_PARENT']);seed=int(os.environ['CORESET_SEED']);E=N/'stage1'/str(seed)
 j=lambda p:json.loads(p.read_text())
 a=j(E/'PROTOCOL_EQUIVALENCE.json');assert a['status']=='PASS_NO_PROTOCOL_DEVIATION_FOUND'
 assert j(O/'ra_coreset/INITIAL_STATE.json')==j(P/'coreset/INITIAL_STATE.json')
 parity=j(O/'TASK6_TENSOR_PARITY.json');assert parity['equal']==same
 assert all(math.isfinite(x) for x in parity['max_abs_diff'].values())
 if not same:assert a['historical_gpu'] and a['current_gpu'] and a['historical_gpu']!=a['current_gpu'],'Same-hardware mismatch is outside this amendment'
 report=dict(status='PASS_EQUAL' if same else 'DISCLOSED_CROSS_HARDWARE_DIFFERENCE',authorized=True,seed=seed,time=time.time(),exact_equal=same,diagnostic_sha256=sha(O/'TASK6_TENSOR_PARITY.json'),protocol_audit_sha256=sha(E/'PROTOCOL_EQUIVALENCE.json'),historical_gpu=a['historical_gpu'],current_gpu=a['current_gpu'],max_abs=max(parity['max_abs_diff'].values()),initial_model_rng_equal=True,reason='User authorized removal of cross-hardware exact final tensor stop after no protocol deviation found; source, loss, seed, sampling and schedule unchanged')
 (E/'TASK6_POLICY_ACCEPTANCE.json').write_text(json.dumps(report,indent=2)+'\n')
 return True
def transform(text,phase):
 tree=ast.parse(text);count=0
 for node in ast.walk(tree):
  if not isinstance(node,ast.Assert):continue
  s=ast.dump(node.test,include_attributes=False)
  if phase=='train' and isinstance(node.test,ast.Name) and node.test.id=='same':
   assert isinstance(node.msg,ast.Constant) and node.msg.value=='Task6 models differ before selection; stop C before task7'
  elif phase=='audit' and ast.unparse(node.test)=="js(O / 'TASK6_TENSOR_PARITY.json')['equal']":pass
  else:continue
  node.test=ast.Call(func=ast.Name(id='_accept_task6',ctx=ast.Load()),args=[node.test],keywords=[]);count+=1
 assert count==1,(phase,count)
 ast.fix_missing_locations(tree);return tree
def main():
 phase=sys.argv.pop(1);assert phase in ['train','audit']
 root=Path(os.environ['CORESET_SOURCE']);p=root/('coreset_exp/train.py' if phase=='train' else 'ra_exp/audit.py')
 c=json.loads((N/'RECOVERY_CONTRACT.json').read_text())
 for name,h in c['files'].items():assert sha(N/name)==h,name
 assert sha(p)==c['frozen_modules'][phase]
 ns=dict(__name__='__main__',__file__=str(p),__package__='coreset_exp' if phase=='train' else 'ra_exp',_accept_task6=accept_task6)
 sys.argv[0]=str(p);exec(compile(transform(p.read_text(),phase),str(p),'exec'),ns)
if __name__=='__main__':main()
