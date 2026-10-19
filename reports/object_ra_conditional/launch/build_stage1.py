"""Build new OBJECT RA namespaces; preserve selector and model/optimizer/eval rules."""
from pathlib import Path
import json,shutil,hashlib,difflib,yaml
R=Path(__file__).resolve().parent;B=R.parent/'20261019_ra_coreset_goal_seed41/ra';S=R/'source'
Q='/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019'
parents={41:('/root/mlr_ifa_new_experiments/Seeds40_41_three_arm_20261016/seed41/rc','/data/mlr_ifa/experiment_workspaces/Seeds40_41_three_arm_20261016/seed41/rc/source'),42:('/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed42_20261013','/data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_seed42_20261013/source'),43:('/data/mlr_ifa/experiments/Seed43_Random_Coreset_repeat_20261016_v1','/data/mlr_ifa/experiment_workspaces/Seed43_three_arm_repeat_20261016_v1/rc/source')}
assert not S.exists();shutil.copytree(B/'source',S,ignore=shutil.ignore_patterns('__pycache__'))
def edit(n,old,new):
 p=S/n;s=p.read_text(encoding='utf-8');assert old in s,(n,old);p.write_text(s.replace(old,new),encoding='utf-8',newline='\n')
edit('coreset_exp/common.py','assert SEED == 41','assert SEED in (41,42,43)')
edit('coreset_exp/train.py',"get_benchmark('LIBERO_GOAL')","get_benchmark('LIBERO_OBJECT')")
edit('coreset_exp/model.py',"c.benchmark_name=='LIBERO_GOAL' and c.gamma==.7 and c.pretrain.endswith('goal_multitask_model_ep96.pth')","c.benchmark_name=='LIBERO_OBJECT' and c.gamma==.3 and c.pretrain.endswith('object_multitask_model_ep99.pth')")
edit('ra_exp/prepare.py','SEED == 41','SEED in (41,42,43)')
edit('ra_exp/prepare.py',"fm['contract_sha256']==sha(P/'CONTRACT.json')","fm['contract_sha256']==pc.get('feature_origin_contract_sha256',sha(P/'CONTRACT.json'))")
edit('ra_exp/prepare.py',"[997,889,904,1969]","[1562,1480,1560,1319]")
edit('ra_exp/prepare.py',"c['gamma']==.7","c['gamma']==.3 and c['benchmark_name']=='LIBERO_OBJECT'")
# OBJECT has known ambiguous constant-action windows. Identity is only diagnostic:
# never invent target time, nor let uncertainty influence the RA selector.
old="""            assert len(identity)==1,('Ambiguous target identity',t,p.name,hits)
            demo,target,length=next(iter(identity));ambiguous+=len(hits)>1
            windows[p.name]=dict(sample_id=p.name,sha256=sha(p),demo_id=demo,target=target,length=length,last_action=a[-1].tolist(),possible_starts=[r['start'] for r in hits],binding='exact complete 8x7 float32 window; unique demo/target/length')"""
new="""            unique=len(identity)==1
            demos={r[0] for r in identity};lengths={r[2] for r in identity}
            demo=next(iter(demos)) if len(demos)==1 else 'AMBIGUOUS'
            target=next(iter(identity))[1] if unique else None
            length=next(iter(lengths)) if len(lengths)==1 else None
            ambiguous+=not unique
            windows[p.name]=dict(sample_id=p.name,sha256=sha(p),demo_id=demo,target=target,length=length,last_action=a[-1].tolist(),possible_starts=[r['start'] for r in hits],candidate_identities=[list(x) for x in sorted(identity)],identity_unique=unique,binding='exact complete 8x7 float32 payload; ambiguous target remains unknown and unused by selection')"""
edit('ra_exp/prepare.py',old,new)
edit('ra_exp/prepare.py','ambiguous_start_but_unique_target=ambiguous','ambiguous_target_identity=ambiguous')
edit('ra_exp/select_offline.py',"unique_times=len({(r['demo_id'],r['target']) for r in rr}),endpoint_windows=sum(r['target']==r['length']-1 for r in rr),demo_support=len({r['demo_id'] for r in rr})","unique_times_known=len({(r['demo_id'],r['target']) for r in rr if r['identity_unique']}),unknown_target_windows=sum(not r['identity_unique'] for r in rr),endpoint_windows_known=sum(r['target']==r['length']-1 for r in rr if r['identity_unique']),demo_support_known=len({r['demo_id'] for r in rr if r['demo_id']!='AMBIGUOUS'})")
for n in ['ra_exp/audit.py','ra_exp/freeze.py']:
 edit(n,'76100','105500')
edit('ra_exp/audit.py','[499,445,452,985]','[781,740,780,660]')
edit('ra_exp/freeze.py','RA_GOAL_extension_legacy_protocol_v1','RA_OBJECT_conditional_legacy_protocol_v1')
edit('ra_exp/freeze.py',"progression=dict(AUC='strictly greater than same-seed Coreset',FWT='not less than same-seed Coreset',OldFinal='not less than same-seed Coreset')","progression=dict(authority='queue DECISION_GRAPH.json',first_comparator='same-seed Random',gate='three-seed exact mean AUC delta positive and at least two positive seed deltas; engineering all PASS')")
edit('ra_exp/audit.py',"criteria=dict(AUC=nm['AUC']>cm['AUC'],FWT=nm['FWT']>=cm['FWT'],OldFinal=nm['OldFinal']>=cm['OldFinal'])","criteria=dict(AUC_vs_Random=nm['AUC']>rm['AUC'])")
proof={'parent':str(B/'source'),'changed':{},'unchanged':{}};patch=[]
for p in sorted(S.rglob('*')):
 if not p.is_file():continue
 n=p.relative_to(S).as_posix();old=B/'source'/n;h=hashlib.sha256(p.read_bytes()).hexdigest()
 if p.read_bytes()==old.read_bytes():proof['unchanged'][n]=h
 else:
  proof['changed'][n]=h;patch.extend(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),p.read_text(encoding='utf-8').splitlines(True),fromfile='RA_GOAL/'+n,tofile='RA_OBJECT/'+n))
 if p.suffix=='.py':compile(p.read_text(encoding='utf-8'),str(p),'exec')
for n in ['ra_exp/selector_reference.py','ra_exp/storage.py','ra_exp/preflight.py','coreset_exp/evaluate.py','coreset_exp/eval_metric.py','libero/lifelong/algos/mlr.py','libero/lifelong/models/brief.py','libero/lifelong/models/complete_policy.py','libero/lifelong/selectors/greedy_kcenter.py']:
 assert n in proof['unchanged'],n
(R/'SOURCE_DIFF.json').write_text(json.dumps(proof,indent=2)+'\n');(R/'source.patch').write_text(''.join(patch),encoding='utf-8',newline='\n')
for seed,(P,PS) in parents.items():
 W=Q+'/seed'+str(seed);O='/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%seed;d=R/('seed%d'%seed);d.mkdir()
 for n in ['pipeline.py','supervise.sh','admission.sh','final_process_audit.py','package_light.py']:
  s=(B/n).read_text(encoding='utf-8').replace('/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed41_three_arm_20261019/ra',W)
  if n=='final_process_audit.py':s=s.replace("status='ELIGIBLE_FOR_EXTENSION' if a['metric_gate'] else 'COMPLETED_METRIC_GATE_NOT_PASSED'","status='COMPLETED_PENDING_THREE_SEED_GATE'")
  (d/n).write_text(s,encoding='utf-8',newline='\n')
 env=f'''#!/usr/bin/env bash
source /data/mlr_ifa/activate.sh
export CORESET_SOURCE={W}/source CORESET_OUT={O} CORESET_SEED={seed}
export RA_PARENT={P} RA_PARENT_SOURCE={PS}
export PYTHONPATH="$CORESET_SOURCE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
cd "$CORESET_SOURCE"
'''
 (d/'env.sh').write_text(env,encoding='utf-8',newline='\n')
 spec=dict(status='PREPARING',suite='LIBERO_OBJECT',seed=seed,arm='ra_coreset',parent=P,parent_source=PS,official_pretrain='/data/mlr_ifa/checkpoints/object_multitask_model_ep99.pth',use_state=True,tasks=[6,7,8,9],batch=10,epochs_each=100,optimizer_steps=105500,final_checkpoints=4,episodes=200,vector_attempts=20,gamma=.3,ifa_weight=.1,warmup500_step_linear=True,windows=[1562,1480,1560,1319],quota=[781,740,780,660],anchors=[390,370,390,330],selector_calibration_seed=42,sampler_seed=0,anchor_training_eval_seed=seed,env_seeds=[seed,seed+1],evaluation='original 0-19 / 10lane / 600 / 5zero / per-stage global RNG / argmax component Normal',selector='unchanged RA half quota anchors minimal geometric removal',identity_ambiguity='explicit diagnostic-only unknown; no guessed target',task6_exact_tensor_parity_required=True,gate='DECISION_GRAPH.json: three seed RA vs Random mean AUC positive; >=2 individual positive; engineering PASS')
 (d/'ACTUAL_SPEC.yaml').write_text(yaml.safe_dump(spec,sort_keys=False),encoding='utf-8',newline='\n')
 (d/'PROTOCOL.md').write_bytes((R/'PROTOCOL.md').read_bytes())
(R/'PARENTS.json').write_text(json.dumps({str(k):dict(root=v[0],source=v[1]) for k,v in parents.items()},indent=2)+'\n')
print(json.dumps({'changed':list(proof['changed']),'unchanged_count':len(proof['unchanged'])},indent=2))
