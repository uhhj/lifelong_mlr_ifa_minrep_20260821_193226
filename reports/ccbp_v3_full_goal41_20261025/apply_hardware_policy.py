"""Predeployment only: user authorized 4080S, historical cross-card equality is descriptive.
No change to loss, data, budgets, FP32, or same-device recovery/arm parity gates.
"""
from pathlib import Path
import ast,json,hashlib,datetime
R=Path(__file__).resolve().parent
changes={}
def edit(name,before,after):
    p=R/name;s=p.read_text(encoding='utf8');assert s.count(before)==1,(name,before[:100])
    changes.setdefault(name,dict(before=hashlib.sha256(p.read_bytes()).hexdigest()))
    s=s.replace(before,after)
    if name.endswith('.py'):ast.parse(s)
    p.write_text(s,encoding='utf8');changes[name]['after']=hashlib.sha256(p.read_bytes()).hexdigest()

edit('train.py',"expected=js(O/'ra_coreset/INITIAL_STATE.json');assert tensor==expected['tensors'] and digest==expected['rng'],'Official initialization/RNG mismatch'",
"""expected=js(O/'ra_coreset/INITIAL_STATE.json')
 assert tensor==expected['tensors'] and all(digest[k]==expected['rng'][k] for k in digest if k!='cuda'),'Official initialization/non-CUDA RNG mismatch'
 props=torch.cuda.get_device_properties(0)
 hardware=dict(device=props.name,capability=[props.major,props.minor],total_memory=props.total_memory,torch_version=torch.__version__,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)
 if arm!='C':
  first=js(R/'INITIAL_C.json');assert tensor==first['tensors'] and digest==first['rng'] and hardware==first['hardware'],'Same-device arm initialization/RNG/hardware mismatch'""")
edit('train.py',"dict(status='PASS_OFFICIAL_INITIALIZATION',tensors=tensor,rng=digest,official_cp_sha256=sha(a.cfg.pretrain),parent_initial_sha256=sha(O/'ra_coreset/INITIAL_STATE.json'))",
"dict(status='PASS_OFFICIAL_INITIALIZATION',tensors=tensor,rng=digest,hardware=hardware,historical_cuda_rng_equal=digest['cuda']==expected['rng']['cuda'],historical_cuda_rng_gate=False,official_cp_sha256=sha(a.cfg.pretrain),parent_initial_sha256=sha(O/'ra_coreset/INITIAL_STATE.json'))")
edit('train.py',"write(R/('TASK6_PARITY_'+arm+'.json'),dict(equal=same,source_sha256=sha(O/'ra_coreset/models/mlrifa_6_model_ep99.pth'),actual_sha256=sha(target)));assert same,'Task6 must reproduce original ALIGN'",
"""actual=cpu(a.policy.state_dict());reference=actual if arm=='C' else torch.load(model_path('C',6),map_location='cpu')['state_dict']
    same_device=equal(actual,reference);assert same_device,'Task6 must match current-card C exactly'
    write(R/('TASK6_PARITY_'+arm+'.json'),dict(historical_ALIGN_equal=same,historical_equality_gate=False,historical_max_abs=max(float((actual[k].double()-baseline[k].double()).abs().max()) for k in actual),same_device_C_equal=same_device,source_sha256=sha(O/'ra_coreset/models/mlrifa_6_model_ep99.pth'),actual_sha256=sha(target),current_C_sha256=sha(model_path('C',6))))""")
edit('audit_progress.py',"assert a['tensors']==parent_initial['tensors'] and a['rng']==parent_initial['rng']",
"assert a['tensors']==parent_initial['tensors'] and all(a['rng'][k]==parent_initial['rng'][k] for k in a['rng'] if k!='cuda')\n  if arm!='C':assert a['tensors']==light['INITIAL_C.json']['tensors'] and a['rng']==light['INITIAL_C.json']['rng'] and a['hardware']==light['INITIAL_C.json']['hardware']")
edit('final_readonly.py',"strict={};diags={};teacher_audits={};calls=0;windows=0;tic=time.time();task6={}","strict={};diags={};teacher_audits={};calls=0;windows=0;tic=time.time();task6={};historical_task6={}")
edit('final_readonly.py',"baseline=torch.load(O/'ra_coreset/models/mlrifa_6_model_ep99.pth',map_location='cpu')['state_dict'];assert all(torch.equal(v,baseline[k]) for k,v in raw['state_dict'].items());continue",
"""baseline=torch.load(O/'ra_coreset/models/mlrifa_6_model_ep99.pth',map_location='cpu')['state_dict']
   historical_task6[arm]=dict(equal=all(torch.equal(v,baseline[k]) for k,v in raw['state_dict'].items()),max_abs=max(float((v.double()-baseline[k].double()).abs().max()) for k,v in raw['state_dict'].items()),equality_gate=False)
   parity=js(Q/('TASK6_PARITY_'+arm+'.json'));assert parity['same_device_C_equal'] and parity['actual_sha256']==sha(cp)
   continue""")
edit('final_readonly.py',"task6_equal_to_ALIGN_and_between_arms=True", "task6_between_current_arms_exact=True,historical_task6=historical_task6")
edit('audit_final_local.py',"assert v['task6_equal_to_ALIGN_and_between_arms']", "assert v['task6_between_current_arms_exact']\nassert set(v['historical_task6'])==set('CRG') and all(x['equality_gate'] is False for x in v['historical_task6'].values())")
edit('audit_final_local.py',"final_cpu=dict(strict_CPs=12,", "hardware_policy='Current-card C/R/G comparisons; historical cross-card equality descriptive only',historical_task6=v['historical_task6'],final_cpu=dict(strict_CPs=12,")
edit('PROTOCOL.md','Task6辅助为零，须实际复现官方ALIGN Task6 tensor。','Task6辅助为零，新卡C/R/G须逐位一致；与旧卡ALIGN Task6 tensor的差异只记录，不作准入否决。官方初始化模型及非CUDA随机状态仍须核验，三组本卡全部初始RNG保持一致。')
edit('PROTOCOL.md','Task6四方（C/R/G/原ALIGN）tensor一致','Task6本卡C/R/G tensor一致及旧卡ALIGN差异记录')
(R/'HARDWARE_POLICY_CHANGE.json').write_text(json.dumps(dict(status='PREDEPLOY_USER_AUTHORIZED_HARDWARE_POLICY',time=datetime.datetime.now().astimezone().isoformat(),user_instruction='开了张4080S，跑的时候忽略掉硬件差异',scientific_loss_input_budget_changed=False,same_card_restore_exact_required=True,files=changes),indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print('Applied predeployment hardware policy; no training/evaluation run.')
