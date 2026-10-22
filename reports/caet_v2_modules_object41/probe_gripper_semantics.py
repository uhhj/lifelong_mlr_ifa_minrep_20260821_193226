"""Read installed execution chain source; no imports/models/environments remotely."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
import ast,json,hashlib
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAETv2_TAC_RSET_EFTR_OBJECT41_20261022');S=Q/'RSET/source';B=Path('/data/mlr_ifa/env/lib/python3.8/site-packages/robosuite')
paths=[S/n for n in ['coreset_exp/evaluate.py','coreset_exp/eval_metric.py','libero/lifelong/models/complete_policy.py','libero/lifelong/models/policy_head.py','libero/libero/envs/venv.py','libero/libero/envs/env_wrapper.py','libero/libero/envs/bddl_base_domain.py','libero/libero/envs/robots/mounted_panda.py','libero/libero/envs/robots/on_the_ground_panda.py']]+[B/n for n in ['environments/base.py','environments/robot_env.py','robots/single_arm.py','robots/manipulator.py','models/robots/manipulators/panda_robot.py','models/grippers/panda_gripper.py','models/grippers/gripper_model.py','models/assets/grippers/panda_gripper.xml']]
out=dict(status='SOURCE_READ',files={},optimizer_calls=0,environment_calls=0,server_writes=0)
for p in paths:
 if not p.exists():out['files'][str(p)]=dict(missing=True);continue
 raw=p.read_bytes();txt=raw.decode();out['files'][str(p)]=dict(sha256=hashlib.sha256(raw).hexdigest(),text=txt)
for n in ('RUN_STATUS.json','controller.exit'):
 p=Q/n;out[n]=p.read_text() if p.exists() else None
import numpy as np,types
txt=(B/'models/grippers/panda_gripper.py').read_text();tree=ast.parse(txt);cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PandaGripper');fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='format_action');scope={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<installed-native-format-action>','exec'),scope)
tests=[]
for current in ([0.,0.],[-1.,1.],[1.,-1.],[-.3,.3]):
 for action in (-2.,-1.,-1e-12,0.,1e-12,1.,2.):
  stub=types.SimpleNamespace(current_action=np.array(current),dof=1,speed=.01);v=scope['format_action'](stub,np.array([action]));expected=np.clip(np.array(current)+np.array([-1.,1.])*.01*np.sign(action),-1.,1.);assert np.array_equal(v,expected)
  tests.append(dict(previous=current,command=action,result=v.tolist(),exact=True))
out['pure_native_formatter_tests']=tests;out['semantics']='negative OPEN, zero HOLD internal gripper target, positive CLOSE; stateful incremental target and saturation, not instantaneous aperture'
print(json.dumps(out))
'''
def main():
 p=R/('gripper_semantics_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir();c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=90);raw=o.read();err=e.read();rc=o.channel.recv_exit_status();(p/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2)+'\n');assert rc==0,err.decode();(p/'SOURCE_EVIDENCE.json').write_bytes(raw)
  print(json.dumps(dict(directory=str(p),sha256=hashlib.sha256(raw).hexdigest(),files={k:('MISSING' if v.get('missing') else v['sha256']) for k,v in json.loads(raw)['files'].items()}),indent=2))
 finally:pw=None;c.close()
if __name__=='__main__':main()
