"""External final-audit adaptation only; never dispatch, train, or evaluate."""
from pathlib import Path
import ast,json,hashlib,difflib
R=Path(__file__).resolve().parent;T=R.parent/'20261026_ccbp_v4_s_object41_42_43'
OLD='CCBP_V4_S_OBJECT41_42_43_20261026_v1';NEW='SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1'
def put(n,s):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
extra='''
 # Independent gate, support, fixed-probe, and physical-cost audit.
 gates=[json.loads(line) for line in (Q/'GATES.jsonl').read_text().splitlines()]
 assert len(gates)==9100
 physical=0;offset=0;by_stage={}
 for stage in range(6,10):
  nb=(POP[stage]+9)//10
  if stage>6:
   pr=js(Q/('PROBES_%d.json'%stage));ids=[i for i,r in enumerate(manifests[stage]) if r['task']==stage and r['bc_valid']]
   gen=np.random.RandomState(20261027+SEED*100+stage)
   assert pr['new_indices']==sorted(int(x) for x in gen.choice(ids,min(64,len(ids)),replace=False))
   yy=torch.stack([torch.as_tensor(labels[stage][i]['next_action']).float() for i in ids])
   sc=torch.stack([yy[:,a:b].square().mean().sqrt().clamp_min(.1) for a,b in [(0,3),(3,6),(6,7)]])
   assert np.allclose(sc.numpy(),pr['new_scale'],atol=1e-7,rtol=1e-7) and pr['new_population']==len(ids)
   assert pr['source']=='TRAINING_ONLY_FIXED_NOT_VALIDATION'
   assert set(pr['old_indices'])==set(pr['support'])==set(pr['support_trajectories'])=={str(t) for t in range(6,stage)}
   for t in range(6,stage):
    rows=cached[t]['rows'];meta=js(Q/('MEMORY_S_%d.json'%t));trajectories=sorted({int(x['trajectory']) for x in rows})
    assert meta['training_attempts']==10 and len(trajectories)==meta['collection_successes']
    assert pr['support'][str(t)]==len(trajectories)/10. and pr['support_trajectories'][str(t)]==trajectories
    gen=np.random.RandomState(20261027+SEED*1000+t)
    expected=sorted(int(x) for x in gen.choice(len(rows),min(8,len(rows)),replace=False)) if rows else []
    assert pr['old_indices'][str(t)]==expected
    binding=js(Q/('TRAIN_INPUT_S_%d.json'%stage))['memory'][str(t)]
    assert binding['support']==len(trajectories)/10. and binding['support_trajectories']==trajectories
  subset=[g for g in gates if g['stage']==stage];accepted_q=0
  for ep in range(100):
   rr=hist[(stage-6)*100+ep];gg=[g for g in subset if g['epoch']==ep]
   assert [g['batch'] for g in gg]==(list(range(0,nb,10)) if stage>6 else [])
   kept=0
   for g in gg:
    step=offset+ep*nb+g['batch'];assert g['start_step']==step and g['steps']==min(10,nb-g['batch']) and g['end_step']==step+g['steps']
    c,q=g['C'],g['Q'];checks=dict(bc_new=q['bc_new']<=c['bc_new'],exec_new=q['exec_new']<=c['exec_new'],retain_improved=q['retain_support']<c['retain_support'])
    assert all(math.isfinite(x[k]) for x in [c,q] for k in ['bc_new','exec_new','retain_support'])
    assert checks==g['checks'] and g['accepted']==('Q' if all(checks.values()) else 'C')
    if g['accepted']=='Q':kept+=g['steps'];accepted_q+=1
   physical+=nb*(2 if stage>6 else 1)
   assert rr['physical_optimizer_calls']==physical and rr['totals']['anchor_slots']==kept*4
  offset+=100*nb
  if stage>6:by_stage[str(stage)]=dict(blocks=len(subset),Q_accepts=accepted_q,support=pr['support'],fixed_probes_verified=True)
 assert physical==195300 and offset==105500
 gate_audit=dict(status='PASS_INDEPENDENT_RECOMPUTATION',blocks=9100,physical_training_optimizer_calls=physical,accepted_updates=offset,by_stage=by_stage)
'''
scripts={};review={}
for seed in (41,42,43):
 n=f'final_seed{seed}_readonly.py';old=(T/n).read_text(encoding='utf8');s=old.replace(OLD,NEW)
 s=s.replace('import os,sys,ast,json,hashlib,time,contextlib','import os,sys,ast,json,hashlib,time,contextlib,math')
 s=s.replace(" rec=torch.load(OUTPUT[arm]/'recovery.pth'",extra+" rec=torch.load(OUTPUT[arm]/'recovery.pth'")
 s=s.replace(" assert rec['memory_bindings']==", " assert rec['physical_optimizer_calls']==195300 and rec['gate_blocks']==9100 and rec['probe_sha256']==sha(Q/'PROBES_9.json')\n assert rec['memory_bindings']==")
 s=s.replace("strict=strict,memory_audits=", "strict=strict,gate_audit=gate_audit,memory_audits=")
 ast.parse(s,feature_version=(3,8));put(n,s);scripts[str(seed)]=s
 put('review/'+n+'.diff',''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='accepted_full_S/'+n,tofile='latest_full_S/'+n)))
 review[n]=dict(parent_sha256=sha(T/n),new_sha256=sha(R/n))
check=(R/'check_readonly.py').read_text(encoding='utf8')
s='''"""Independent CPU final audit. Rejects unfinished or live experiments."""
import subprocess,json,time
'''+ 'CHECK_SCRIPT='+repr(check)+'\nSCRIPTS='+repr(scripts)+'''\n
def run(script):
 p=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-'],input=script,text=True,capture_output=True)
 assert p.returncode==0,p.stderr
 return json.loads(p.stdout)
v=run(CHECK_SCRIPT)
assert v['light'].get('controller.exit',{}).get('text','').strip()=='0' and not v['processes'] and not v['canceled_mechanism_processes']
results={};diagnostics={}
for seed,script in SCRIPTS.items():
 r=run(script)
 assert r['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and r['contract_sha256']==v['children'][seed]
 for k,rows in r.pop('diagnostics').items():diagnostics['s'+seed+'_'+k]=rows
 for n,x in r.pop('light').items():
  if 'seed'+seed+'/'+n in v['light']:assert x['sha256']==v['light']['seed'+seed+'/'+n]['sha256']
 results[seed]=r
after=run(CHECK_SCRIPT)
assert after['contract_sha256']==v['contract_sha256'] and after['light']==v['light'] and not after['processes']
v.update(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',observed=time.time(),seeds=results,diagnostics=diagnostics)
print(json.dumps(v))
'''
ast.parse(s,feature_version=(3,8));put('final_readonly.py',s)
put('audit_seed_evaluation.py',(T/'audit_seed_evaluation.py').read_text(encoding='utf8'))
s=(T/'audit_final_local.py').read_text(encoding='utf8')
s=s.replace("a['total_epochs']", "a['epochs']").replace("a['total_main_updates']", "a['accepted_updates']").replace("a['isolated_optimizer_updates']==66", "a['own_isolated_optimizer_calls']==273 and a['own_extra_autograd_calls']==123 and a['physical_training_optimizer_calls']==585900").replace("a['training_collection_episodes']", "a['training_collection_completed_episodes']")
s=s.replace("progress=a['seeds'][key];ev=a['evaluations'][key]",'''progress=a['counts'][key]
 child=D/('final_seed'+key);child.mkdir(exist_ok=True);light=child/'light';light.mkdir(exist_ok=True)
 for n,x in r['light'].items():
  fp=light/n;fp.parent.mkdir(parents=True,exist_ok=True);fp.write_text(x['text'],encoding='utf8')
 (child/'EVIDENCE.json').write_text(json.dumps(dict(observed=v['observed'])),encoding='utf8')
 (child/'LOCAL_AUDIT.json').write_text(json.dumps(dict(failure=None)),encoding='utf8')
 subprocess.run([sys.executable,'-X','utf8',str(R/'audit_seed_evaluation.py'),str(child),key],check=True)
 ev=js(child/'EVALUATION_PROGRESS_AUDIT.json')''')
s=s.replace("['gradient_autograd_calls']==72", "['gradient_autograd_calls']==36").replace("for x in gg)==72", "for x in gg)==36")
old="  for term in ['bc','ifa_weighted','expert_weighted','expert_route_weighted','retain_weighted','retain_route_weighted']:assert np.isfinite([x[term]['loss'],x[term]['l2']]).all() and x[term]['l2']>=0"
new="""  assert x['extra_autograd_calls']==3 and np.isfinite([x['bc_new'],x['exec_new'],x['retain_weighted']]).all()
  assert all(math.isfinite(y) and y>=0 for y in x['norms'].values())
  for k in ['retain_vs_bc','retain_vs_exec']:
   assert math.isfinite(x[k]['dot']) and (x[k]['cosine'] is None or math.isfinite(x[k]['cosine']) and abs(x[k]['cosine'])<=1.000001)"""
assert old in s;s=s.replace(old,new)
s=s.replace('isolated_optimizer_updates=66,formal_extra_autograd=216,admission_extra_autograd=78', 'physical_training_optimizer_calls=585900,isolated_optimizer_updates=273,formal_extra_autograd=108,admission_extra_autograd=123')
s=s.replace('results[key]=dict(matrix=matrix',"assert r['gate_audit']['status']=='PASS_INDEPENDENT_RECOMPUTATION' and r['gate_audit']['blocks']==9100 and r['gate_audit']['physical_training_optimizer_calls']==195300\n results[key]=dict(gate_audit=r['gate_audit'],matrix=matrix")
s=s.replace('Only S OBJECT41/42/43 added; no C or O controlled OBJECT arms.', 'Only latest support-gated S OBJECT41/42/43; the canceled mechanism experiment is not resumed or required for acceptance.')
s=s.replace('Historical baselines have different action RNG and no extra online training collection; not equal-interaction comparisons.', 'Historical methods are read-only references; extra acquisition and double candidates prohibit equal-interaction or equal-compute causal claims.')
ast.parse(s,feature_version=(3,8));put('audit_final_local.py',s)
put('FINAL_WRAPPER_REVIEW.json',json.dumps(dict(status='STATIC_ONLY_NOT_EXECUTED_FINAL_ACCEPTANCE_PENDING',sources=review,generated={n:sha(R/n) for n in ['final_readonly.py','audit_final_local.py','audit_seed_evaluation.py']},actual_final_CPU_checks=0,training_updates=0,environment_calls=0,requires='Natural controller.exit 0, no own processes, then actual CPU checks and local acceptance; frozen scientific sources unchanged'),indent=2)+'\n')
print('External final wrappers adapted and Python3.8 parsed; no remote execution or acceptance.')
