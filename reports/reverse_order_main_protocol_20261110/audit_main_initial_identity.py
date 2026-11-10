"""Bind the frozen initialization gate directly to each actual main-table MLR run."""
from pathlib import Path
import datetime,hashlib,json
R=Path(__file__).resolve().parent
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
results={}
for key,v in J(R/'MAIN_TABLE_BINDING.json')['cases'].items():
    suite,seed=key.split('_');p=Path(v['MLR']['actual_result_root'])/'random/INITIAL_STATE.json'
    old=J(p);inp=J(R/('cases/%s_MLR_%s/INPUT.json'%(suite,seed)))['initial_state']
    assert old['tensors']==inp['tensors'] and old['rng']==inp['rng']
    results[key]=dict(main_initial_path=str(p),main_initial_sha256=sha(p),frozen_reverse_input_sha256=sha(R/('cases/%s_MLR_%s/INPUT.json'%(suite,seed))),all100_tensor_hashes_identical=True,all_python_numpy_cpu_and_recorded_cuda_rng_hashes_identical=True)
out=dict(status='PASS_READONLY_MAIN_INITIAL_IDENTITY_NOT_NEW_GPU_ADMISSION',observed=datetime.datetime.now().astimezone().isoformat(),cases=results,new_model_forward=0,new_optimizer=0,new_environment=0,frozen_inputs_changed=False)
(R/'MAIN_INITIAL_IDENTITY_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=out['status'],cases=len(results),scientific_calls=0)))
