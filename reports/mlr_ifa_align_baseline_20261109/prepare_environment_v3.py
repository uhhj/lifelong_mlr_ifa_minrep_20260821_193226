"""Versioned zero-science startup repair, preserving failed v1 and all inputs."""
from pathlib import Path
import json,hashlib,ast,difflib
R=Path(__file__).resolve().parent;OLD=R/'engineering_completion_gate_20261110_v2';D=R/'engineering_environment_20261110_v3'
assert not D.exists();D.mkdir()
sha=lambda t:hashlib.sha256(t.encode()).hexdigest()
old=json.loads((OLD/'DEPLOY_TEXT_BUNDLE.json').read_text(encoding='utf8'))
oldname='MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1';name='MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2'
b={n:t.replace(oldname,name) for n,t in old.items()}
failure=json.loads((OLD/'capacity_readonly_20261110_122814/EVIDENCE.json').read_text(encoding='utf8'))
assert failure['status']=='PASS_FAILED_IMPORT_BEFORE_INITIALIZATION_ZERO_SCIENCE'
runtime=b['runtime.py']
needle=" assert not processes(roots),'Current q=1 still active; never interrupt it'"
extra="""
 failed=Q.parent/'MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1'
 failed_private=P.parent/failed.name
 assert sha(failed/'CONTRACT.json')=='db88ef7991d4cfd6e77b6897977779ed5d37a85682cd63c14441bb63bddcd7c3'
 assert (failed/'controller.exit').read_text().strip()=='1'
 assert sha(failed/'FAILURE.json')=='9457a2c32359c38ade6b5811633eb447f9647b7e8070ea739399779978e48c82'
 assert not processes([failed,failed_private])
 config=Path('/data/mlr_ifa/config/config_adaptation.yaml')
 assert sha(config)=='44ee53d32cdb509569ca7fb6b2674ce81f51deb95908e9abde093ff3677f72bf'
"""
assert needle in runtime;runtime=runtime.replace(needle,needle+extra)
oldenv="CUDA_VISIBLE_DEVICES='0')"
newenv="CUDA_VISIBLE_DEVICES='0',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',MUJOCO_GL='egl',MPLBACKEND='Agg',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')"
assert runtime.count(oldenv)==1;runtime=runtime.replace(oldenv,newenv);b['runtime.py']=runtime
# Rebind only output locations and corresponding metadata hashes.
for case in json.loads(old['CONTRACT.json'])['cases']:
 n='private/'+case+'/CONTRACT.json';c=json.loads(b[n]);oc=json.loads(old[n])
 assert c['source_sha256']==oc['source_sha256']
 for path in list(c['files']):
  prefix='/data/mlr_ifa/experiments/'+name+'/'
  if path.startswith(prefix):c['files'][path]=sha(b['private/'+path[len(prefix):]])
 b[n]=json.dumps(c,indent=2)+'\n'
 for k,h in c['source_sha256'].items():assert sha(b['cases/'+case+'/source/'+k])==h
 for path,h in c['files'].items():
  if name not in path:assert oc['files'][path]==h
 for n0 in ['INPUT.json','resolved_eval.json','resolved_random.json']:
  assert b['private/'+case+'/'+n0].replace(name,oldname)==old['private/'+case+'/'+n0]
c=json.loads(b['CONTRACT.json'])
c['code_files']={n:sha(b[n]) for n in c['code_files']}
c['private_files']={n:sha(b['private/'+n]) for n in c['private_files']}
b['CONTRACT.json']=json.dumps(c,indent=2)+'\n';contract=sha(b['CONTRACT.json'])
assert all(old[n]==b[n] for n in old if n.startswith('cases/'))
for n,t in b.items():
 if n.endswith('.py'):ast.parse(t,feature_version=(3,8))
(D/'DEPLOY_TEXT_BUNDLE.json').write_text(json.dumps(b),encoding='utf8')
(D/'PREPARED_CONTRACT.json').write_text(b['CONTRACT.json'],encoding='utf8')
(D/'runtime.py').write_text(runtime,encoding='utf8')
(D/'deploy_remote.py').write_text(b['deploy_remote.py'],encoding='utf8')
(D/'runtime.diff').write_text(''.join(difflib.unified_diff(old['runtime.py'].splitlines(True),runtime.splitlines(True),fromfile='failed_v1/runtime.py',tofile='environment_v2/runtime.py')),encoding='utf8')
for n in ['transport.py','launch_after_q1.py']:(D/n).write_text((OLD/n).read_text(encoding='utf8'),encoding='utf8')
for n in ['check_readonly.py','final_readonly.py','record_startup.py']:
 t=(OLD/n).read_text(encoding='utf8').replace(oldname,name).replace(sha(old['CONTRACT.json']),contract).replace('engineering_completion_gate_20261110_v2','engineering_environment_20261110_v3')
 (D/n).write_text(t,encoding='utf8')
review=dict(status='PASS_STATIC_ZERO_SCIENCE_IMPORT_REPAIR_NOT_DEPLOYED',adapted_contract=contract,failed_root=oldname,active_candidate=name,all_case_training_and_evaluation_sources_byte_exact=True,all_nonoutput_metadata_and_external_inputs_exact=True,changes=['Versioned output paths and dependent metadata hashes','Bind existing LIBERO_CONFIG_PATH/HF_HOME/EGL and single-thread numerical runtime','Require preserved failed-root exit1/hash and no processes'],previous_failure=failure,optimizer_updates=0,environments=0)
(D/'REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in review.items() if k!='previous_failure'},ensure_ascii=False))
