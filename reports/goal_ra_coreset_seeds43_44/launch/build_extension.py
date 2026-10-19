"""Prepare isolated seed43/44 source. Only seed/provenance plumbing changes."""
from pathlib import Path
import json,hashlib,shutil,difflib,zipfile,yaml
R=Path(__file__).resolve().parent;B=R.parent/'20261018_ra_coreset_goal_seed42'
Q='/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_extension_20261019'
S=R/'source';assert not S.exists();shutil.copytree(B/'source',S,ignore=shutil.ignore_patterns('__pycache__'))
def edit(n,old,new):
 p=S/n;s=p.read_text(encoding='utf-8');assert old in s,n;p.write_text(s.replace(old,new),encoding='utf-8',newline='\n')
edit('ra_exp/audit.py','from coreset_exp.common import O,verify,write,sha,hashes','from coreset_exp.common import O,SEED,verify,write,sha,hashes')
edit('ra_exp/audit.py',"r['seed']==42","r['seed']==SEED")
edit('ra_exp/freeze.py','from coreset_exp.common import O,S,sha,write,source_hashes','from coreset_exp.common import O,S,SEED,sha,write,source_hashes')
edit('ra_exp/freeze.py',"version='RA_GOAL_seed42_legacy_protocol_v1',seed=42","version='RA_GOAL_extension_legacy_protocol_v1',seed=SEED")
edit('ra_exp/freeze.py', "    files.update(pc['inputs']);files.update(binding['protected'])", "    files.update(pc['inputs']);files.update(binding['protected'])\n    queue=S.parent.parent;orchestration=json.loads((queue/'ORCHESTRATION_SHA256.json').read_text())\n    for name,h in orchestration.items():assert sha(queue/name)==h,name\n    files.update({str(queue/name):h for name,h in orchestration.items()})\n    files[str(queue/'ORCHESTRATION_SHA256.json')]=sha(queue/'ORCHESTRATION_SHA256.json')")
edit('ra_exp/select_offline.py','from coreset_exp.common import O,sha,write,hashes,rng','from coreset_exp.common import O,SEED,SELECTOR_SEED,sha,write,hashes,rng')
edit('ra_exp/select_offline.py','random.Random(42).sample(ids,q)','random.Random(SEED).sample(ids,q)')
edit('ra_exp/select_offline.py','population,42)','population,SELECTOR_SEED)')
p=S/'ra_exp/prepare.py';s=p.read_text()
s=s.replace('import json,os,shutil,time','import json,os,shutil,time,collections\nimport numpy as np,torch,h5py').replace('from coreset_exp.common import O,sha,write,hashes','from coreset_exp.common import O,SEED,sha,write,hashes')
s=s.replace("    assert json.loads((P/'INDEPENDENT_ACCEPTANCE.json').read_text())['status']=='PASS'", "    assert SEED in (43,44) and pc['seed']==SEED\n    assert json.loads((P/'INDEPENDENT_ACCEPTANCE.json').read_text())['status']=='PASS'")
a=s.index("    BS=Path(os.environ['SC_BINDING_SOURCE'])");b=s.index('    for name,src',a)
s=s[:a]+'''    # Seed-specific cached windows must be bound to their own actual payloads.
    tasks=json.loads((P/'TASKS.json').read_text());bindings={};torch.set_num_threads(1)
    for t in range(6,10):
        hdf=tasks[str(t)]['paths'][0];candidates=collections.defaultdict(list)
        with h5py.File(hdf) as h:
            for demo in fm['demos'][str(t)]:
                a=h['data'][demo]['actions'][:].astype(np.float32)
                for start in range(len(a)):
                    ix=np.minimum(np.arange(start,start+8),len(a)-1)
                    candidates[a[ix].tobytes()].append(dict(demo_id=demo,start=start,target=int(ix[-1]),length=len(a)))
        windows={};ambiguous=0
        for p in sorted((P/'features'/str(t)).glob('*.pt')):
            a=torch.load(p,map_location='cpu')['actions'][0].numpy().astype(np.float32)
            hits=candidates[a.tobytes()];assert hits,(t,p.name)
            identity={(r['demo_id'],r['target'],r['length']) for r in hits}
            assert len(identity)==1,('Ambiguous target identity',t,p.name,hits)
            demo,target,length=next(iter(identity));ambiguous+=len(hits)>1
            windows[p.name]=dict(sample_id=p.name,sha256=sha(p),demo_id=demo,target=target,length=length,last_action=a[-1].tolist(),possible_starts=[r['start'] for r in hits],binding='exact complete 8x7 float32 window; unique demo/target/length')
        assert len(windows)==[997,889,904,1969][t-6]
        write(O/'bindings'/f'task{t}.json',dict(task=t,hdf5=hdf,hdf5_sha256=sha(hdf),windows=windows,ambiguous_start_but_unique_target=ambiguous))
''' +s[b:]
s=s.replace("        c.update(run_root", "        assert c['seed']==SEED and c['gamma']==.7 and c['policy']['use_state'] is True\n        c.update(run_root")
p.write_text(s,encoding='utf-8',newline='\n')
changed={};same={};patch=[]
for p in sorted(S.rglob('*')):
 if not p.is_file():continue
 n=p.relative_to(S).as_posix();old=B/'source'/n;h=hashlib.sha256(p.read_bytes()).hexdigest()
 if p.read_bytes()==old.read_bytes():same[n]=h
 else:
  changed[n]=h;patch.extend(difflib.unified_diff(old.read_text().splitlines(True),p.read_text().splitlines(True),fromfile='seed42/'+n,tofile='extension/'+n))
assert set(changed)=={'ra_exp/audit.py','ra_exp/freeze.py','ra_exp/select_offline.py','ra_exp/prepare.py'}
(R/'SOURCE_DIFF.json').write_text(json.dumps(dict(changed=changed,unchanged=same,algorithm_model_train_eval_unchanged=True),indent=2)+'\n')
(R/'source.patch').write_text(''.join(patch),encoding='utf-8',newline='\n')
for seed in (43,44):
 W=Q+'/seed'+str(seed);O='/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed%d_20261019'%seed
 d=R/('seed'+str(seed));d.mkdir()
 for n in ['pipeline.py','final_process_audit.py','monitor_snapshot.py','package_light.py','supervise.sh','admission.sh']:
  text=(B/n).read_text(encoding='utf-8').replace('/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018',W)
  (d/n).write_text(text,encoding='utf-8',newline='\n')
 text=(B/'env.sh').read_text().replace('/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018',W).replace('/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018',O)
 text=text.replace('CORESET_SEED=42','CORESET_SEED='+str(seed)).replace('Coreset_paper_GOAL_seed42_20261017','Coreset_paper_GOAL_seed%d_20261017'%seed).replace('three_seed_20261017/seed42/source','three_seed_20261017/seed%d/source'%seed)
 text='\n'.join(x for x in text.splitlines() if 'SC_BINDING_SOURCE=' not in x)+'\n'
 (d/'env.sh').write_text(text,encoding='utf-8',newline='\n')
 spec=yaml.safe_load((B/'ACTUAL_SPEC.yaml').read_text())
 spec['status']='PREPARING_EXPLICIT_SEED_EXTENSION'
 spec['source']['GOAL_parent']=f'/root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/seed{seed}/source'
 spec['study']['first_seed']=seed;spec['study']['seed_extension_if_accepted']=[]
 spec['acceptance']['first_seed_success_gate']={'AUC':'RA > corresponding same-seed Coreset','FWT':'RA >= corresponding same-seed Coreset','OldFinal':'RA >= corresponding same-seed Coreset','engineering':'PASS'}
 spec['actual_evaluation']['stage_seed']=seed;spec['actual_evaluation']['env_seeds']=[seed,seed+1]
 spec['extension']={'authorized_seeds':[43,44],'seed':seed,'anchor_seed':seed,'selector_calibration_seed':42,'sampler_seed':0,'run_both_independent_of_metrics':True,'baseline_retraining':False,'baseline_reevaluation':False,'seed42_protocol_reference':'a frozen unchanged completed experiment'}
 (d/'ACTUAL_SPEC.yaml').write_text(yaml.safe_dump(spec,sort_keys=False),encoding='utf-8',newline='\n')
 (d/'PROTOCOL.md').write_bytes((R/'EXTENSION_PROTOCOL.md').read_bytes())
print('Built four plumbing changes; algorithm, model, training, evaluation byte-identical to RA42')
