from pathlib import Path
import json,datetime
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261110_203911'
E=json.loads((D/'EVIDENCE.json').read_text());L=E['light']
def j(k):return json.loads(L[k]['text'])
a=j('OBJECT_ALIGN_41/INITIAL_A.json');b=j('OBJECT_MLR_41/INITIAL_A.json')
assert a['tensors']==b['tensors'] and a['rng']==b['rng'] and a['hardware']==b['hardware']
x=j('OBJECT_ALIGN_41/EPOCHS_A.json');y=j('OBJECT_MLR_41/EPOCHS_A.json')
assert [(z['sampler_sha256'],z['lr_first'],z['lr_after'],z['batches']) for z in x]==[(z['sampler_sha256'],z['lr_first'],z['lr_after'],z['batches']) for z in y]
for variant in ['ALIGN','MLR']:
 ad=j('OBJECT_'+variant+'_41/ADMISSION_A_6.json');assert ad['status']=='PASS_OWN_GPU_STRIPPED_S_ADMISSION' and ad['optimizer_calls']==5
 inp=j('OBJECT_'+variant+'_41/TRAIN_INPUT_A_6.json');assert inp['objective']==('ALIGN_BC' if variant=='ALIGN' else 'ORIGINAL_BC')+' + 0.1*IFA'
row=dict(status='PASS_ACTUAL_STARTUP_NOT_FINAL',observed=datetime.datetime.fromtimestamp(E['observed']).astimezone().isoformat(),evidence=D.name,contract_sha256=E['contract_sha256'],paired_initial_tensors_and_all_rng_exact=True,paired_actual_hardware_identical=a['hardware'],first_6_epochs_sampler_and_lr_identical=True,first_stage_both_admissions_passed=True,future_stage_admission_not_claimed=True,production_epochs_each=6,production_calls_each=942,isolated_admission_calls_each=5,formal=0,training_collection=0,gpu_snapshot=E['gpu'])
(R/'STARTUP_AUDIT.json').write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8')
dep=json.loads((R/'deploy_remote_20261110_203729/EVIDENCE.json').read_text());dep.update(status='ACTIVE_ACTUAL_FIRST_STAGE_ADMISSION_PASSED',startup_evidence=D.name)
(R/'ACTIVE_ROOT.json').write_text(json.dumps(dep,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in row.items() if k!='paired_actual_hardware_identical'}))
