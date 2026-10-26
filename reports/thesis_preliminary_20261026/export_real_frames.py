"""Export six user-requested RGB evidence frames only, no NPZ or policy payload."""
from pathlib import Path
import sys, json, hashlib, base64, datetime
import paramiko
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'20261026_ccbp_v4_s_goal42_43'))
from credential_store import load
CODE=r'''
from pathlib import Path
import numpy as np, json, hashlib, io, base64
from PIL import Image
Q=Path('/data/mlr_ifa/experiment_workspaces/RA_ALIGN_PAIRED_GOAL_20261024_v1_engineering02')
P=Path('/root/mlr_ifa/experiments/RA_ALIGN_PAIRED_GOAL_20261024_v1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'CONTRACT.json')=='12fc5c69ebacbbb0362d089122c5e07f230c8ed6d17294e34f3073bbe8765e67'
contract=json.loads((Q/'CONTRACT.json').read_text())
assert all(sha(Q/n)==h for n,h in contract['files'].items())
assert (Q/'worker.exit').read_text().strip()=='0'
run='ALIGN_s41_driver9_g0';idx=3
out=next(x for x in json.loads((Q/'OUTCOMES.json').read_text()) if x['run']==run and x['initial_index']==idx)
assert out['success'] is True
manifest=json.loads((Q/'light'/run/'PRIVATE_MANIFEST.json').read_text())
result={'run':run,'suite':'GOAL','task':8,'task_language':'put the bowl on the plate','seed':41,'stage':9,'initial_index':idx,'success':out['success'],'policy_steps':out['policy_steps'],'source_contract_sha256':sha(Q/'CONTRACT.json'),'selection':'Smallest successful ALIGN stage9 seed41 index in previously reviewed records; illustrative successful example, not a success-rate estimate.','images':[],'new_rollouts':0,'model_forward':0,'optimizer_updates':0,'server_writes':0}
for window in ['first_mean_delta','first_mode_switch','terminal']:
 fn='%02d_%s.npz'%(idx,window);p=P/run/fn
 assert sha(p)==manifest[fn]['sha256']
 z=np.load(p,allow_pickle=False)
 end=out['policy_steps'] if window=='terminal' else out['first_events'][window[6:]]
 for camera in ['agentview_image','robot0_eye_in_hand_image']:
  a=z[camera][-1];assert a.dtype==np.uint8 and a.shape==(128,128,3)
  b=io.BytesIO();Image.fromarray(a).save(b,format='PNG');raw=b.getvalue()
  result['images'].append({'name':window+'_'+camera+'.png','camera':camera,'step':end,'source_file':str(p),'source_sha256':sha(p),'frame_index':len(z[camera])-1,'rgb_array_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'png_sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'base64':base64.b64encode(raw).decode()})
print(json.dumps(result))
'''
def main():
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
 secret=load();assert secret,'Saved credential unavailable'
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=secret,allow_agent=False,look_for_keys=False,timeout=20,banner_timeout=20,auth_timeout=20);secret=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -B -',timeout=120)
  i.write(CODE);i.flush();i.channel.shutdown_write();raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();assert rc==0,err
  d=json.loads(raw);dest=R/'assets_private';dest.mkdir(exist_ok=True)
  for row in d['images']:
   png=base64.b64decode(row.pop('base64'));assert hashlib.sha256(png).hexdigest()==row['png_sha256'];(dest/row['name']).write_bytes(png)
  d['export_time']=datetime.datetime.now().astimezone().isoformat();d['transport_script_sha256']=hashlib.sha256(CODE.encode()).hexdigest();d['publication']='PNG for user thesis/deck only; raw NPZ not downloaded; images excluded from Git publication.'
  (R/'REAL_FRAME_PROVENANCE.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
  print(json.dumps({'status':'EXPORTED_HASH_VERIFIED_RGB_ONLY','images':len(d['images']),'steps':[x['step'] for x in d['images']],'new_environments':0}))
 finally:secret=None;c.close()
if __name__=='__main__':main()
