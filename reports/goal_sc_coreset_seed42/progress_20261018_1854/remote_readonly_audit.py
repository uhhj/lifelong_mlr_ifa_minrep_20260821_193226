import os,json,time,subprocess,shutil
from pathlib import Path
from coreset_exp.common import O,verify,sha,hashes,write
P=Path(os.environ['SC_PARENT']);root=O/'sc_coreset'
start=time.time();verify()
b=json.loads((O/'PARENT_BINDING.json').read_text())
assert all(sha(p)==h for p,h in b['protected'].items())
assert hashes(root/'buffer')==json.loads((root/'buffer_after_task9.json').read_text())['hashes']
out=O/'progress_20261018_1854';out.mkdir(exist_ok=False)
files=['commands.jsonl','timings.jsonl','RUN_STATUS.json','TASK6_TENSOR_PARITY.json','train.exit','evaluate.exit','audit.exit','pipeline.exit','sc_coreset/epochs.json','sc_coreset/TRAINING_COMPLETE.json']
files+= [str(p.relative_to(O)) for p in (root/'evaluation').glob('*') if p.suffix in ['.json','.jsonl']]
manifest={}
for rel in files:
 p=O/rel
 if not p.exists():continue
 raw=p.read_bytes()
 if p.suffix=='.jsonl' and raw and not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
 dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
 manifest[rel]={'bytes':len(raw),'sha256':sha(dst)}
snapshot=json.loads(subprocess.check_output(['/data/mlr_ifa/env/bin/python',str(Path(os.environ['CORESET_SOURCE']).parent/'monitor_snapshot.py')]))
check={'status':'PARTIAL','time':time.time(),'frozen_source_config_input_feature_modal':'PASS','parent_inputs':'PASS','actual_buffer_payloads':'PASS','verify_seconds':time.time()-start,'snapshot':snapshot,'storage':{}}
for t in range(6,10):
 m=json.loads((root/'storage'/('task%s.json'%t)).read_text());r=m['selection']
 pre=json.loads((O/'selection_preflight'/('task%s.json'%t)).read_text())
 assert r['selected_ids']==pre['selected_ids'] and r['original_ids']==pre['original_ids']
 check['storage'][str(t)]={'manifest_sha256':sha(root/'storage'/('task%s.json'%t)),'quota':m['K'],'minimum_swaps':r['minimum_swaps'],'exact_preflight_order_match':True}
write(out/'READONLY_AUDIT.json',check)
manifest['READONLY_AUDIT.json']={'bytes':(out/'READONLY_AUDIT.json').stat().st_size,'sha256':sha(out/'READONLY_AUDIT.json')}
write(out/'PACKAGE_MANIFEST.json',{'status':'PARTIAL','time':time.time(),'files':manifest})
print(json.dumps(check))

