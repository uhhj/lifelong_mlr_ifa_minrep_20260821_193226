import json,time,subprocess,os
from pathlib import Path
from coreset_exp.common import O,verify,sha,hashes,write
P=Path(os.environ['SC_PARENT'])
root=O/'sc_coreset'
start=time.time()
c=verify()
b=json.loads((O/'PARENT_BINDING.json').read_text())
assert all(sha(p)==h for p,h in b['protected'].items())
assert hashes(root/'buffer')==json.loads((root/'buffer_after_task9.json').read_text())['hashes']
out={'status':'PARTIAL','observed_unix':time.time(),'frozen_source_inputs_features_modal':'PASS','parent_protected_inputs':'PASS','actual_buffer_payload_hashes':'PASS','seconds':time.time()-start,'checkpoint_metadata':[],'ifa':{}}
for t in range(6,10):
 p=root/'models'/('mlrifa_%s_model_ep99.pth'%t)
 out['checkpoint_metadata'].append({'task':t,'bytes':p.stat().st_size,'sha256':sha(p),'strict_load':'PENDING_FINAL_AUDIT'})
 a=json.loads((root/('ifa_task%s.json'%t)).read_text()); old=json.loads((P/'coreset'/('ifa_task%s.json'%t)).read_text())
 out['ifa'][str(t)]={'current':a,'parent':old,'entire_record_equal':a==old}
out['snapshot']=json.loads(subprocess.check_output(['/data/mlr_ifa/env/bin/python',str(Path(os.environ['CORESET_SOURCE']).parent/'monitor_snapshot.py')]))
write(O/'progress_20261018_1804.json',out)
print(json.dumps({k:v for k,v in out.items() if k not in ['ifa','checkpoint_metadata']}))

