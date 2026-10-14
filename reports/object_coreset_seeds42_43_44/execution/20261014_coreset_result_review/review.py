import json, hashlib, statistics, itertools
from pathlib import Path
from fractions import Fraction as F
P=Path(__file__).resolve().parent
R=P.parent/'20261013_coreset_extension/final_evidence/remote'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(R/'FINAL_DOWNLOAD_MANIFEST.json')
assert all(sha(R/p)==h for p,h in manifest.items())
out={'input_files':len(manifest),'input_hashes_before':'PASS','seeds':{}}
allcells=[];ifas=[];storage=[];cost=[];config_diffs={}
for seed in (42,43,44):
 root=R/f'seed{seed}'; summary=read(root/'SUMMARY.json'); arms={};raw={}
 assert read(root/'INDEPENDENT_ACCEPTANCE.json')['status']=='PASS'
 assert read(root/'TASK6_TENSOR_PARITY.json')['equal']
 assert (root/'random/INITIAL_STATE.json').read_bytes()==(root/'coreset/INITIAL_STATE.json').read_bytes()
 assert all((root/n).read_text().strip()=='0' for n in ('pipeline.exit','supervisor.exit'))
 for arm in ('random','coreset'):
  a=root/arm; ep=read(a/'epochs.json')
  assert [(r['task'],r['epoch']) for r in ep]==[(t,e) for t in range(6,10) for e in range(100)]
  assert ep[-1]['steps']==105500
  cells={};raw[arm]={}
  for i in range(6,10):
   for j in range(6,i+1):
    rows=[json.loads(x) for x in (a/'evaluation'/f'trials_stage{i}_task{j}.jsonl').read_text().splitlines()]
    rows=sorted(rows,key=lambda r:r['trial']);assert [r['trial'] for r in rows]==list(range(20))
    assert all(r['seed']==seed and r['arm']==arm and r['stage']==i and r['task']==j and r['epoch']==99 and type(r['success'])==bool for r in rows)
    events=[json.loads(x) for x in (a/'evaluation'/f'attempts_stage{i}_task{j}.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events]==['environment_construct_start','environment_construct_complete','vector_attempt_start','vector_attempt_complete','vector_attempt_start','vector_attempt_complete','environment_closed']
    assert [e['env_seed'] for e in events if e['event']=='vector_attempt_start']==[seed,seed+1]
    for ev in [e for e in events if e['event']=='vector_attempt_complete']:
     assert ev['successes']==[r['success'] for r in rows][ev['group']*10:(ev['group']+1)*10]
    raw[arm][i,j]=rows;cells[i,j]=F(sum(r['success'] for r in rows),20)
  metrics={'AUC':sum(sum(cells[i,j] for i in range(j,10))/(10-j) for j in range(6,10))/4,
   'FWT':sum(cells[j,j] for j in range(6,10))/4,
   'NBT':sum(sum(cells[j,j]-cells[i,j] for i in range(j+1,10))/(9-j) for j in range(6,9))/3}
  assert all(str(v)==summary['arms'][arm]['metrics'][k]['fraction'] for k,v in metrics.items())
  arms[arm]={'metrics':{k:float(v*100) for k,v in metrics.items()},'matrix':[[float(cells[i,j]*100) if (i,j) in cells else None for j in range(6,10)] for i in range(6,10)]}
  for t in range(6,10):
   info=read(a/f'ifa_task{t}.json'); ifas.append({'seed':seed,'arm':arm,'task':t,'pairs':info['info']['selected_pairs'],'reference_max_deviation':info['reference_max_deviation']})
  cost.append({'seed':seed,'arm':arm,'training_epoch_seconds':sum(r['seconds'] for r in ep),'storage_seconds':sum(read(a/f'buffer_after_task{t}.json')['seconds'] for t in range(6,10))})
 task_contribution={};new_contribution=F(0);old_contribution=F(0)
 for j in range(6,10):
  ds=[]
  for i in range(j,10):
   r=raw['random'][i,j];c=raw['coreset'][i,j]
   assert all(x['initial_state_sha256']==y['initial_state_sha256'] for x,y in zip(r,c))
   counts={k:0 for k in ('SS','SF','FS','FF')}
   for x,y in zip(r,c): counts[('S' if x['success'] else 'F')+('S' if y['success'] else 'F')]+=1
   delta=F(counts['FS']-counts['SF'],20);assert sum(counts.values())==20
   contribution=delta/F(4*(10-j));ds.append(delta)
   if i==j:new_contribution+=contribution
   else:old_contribution+=contribution
   allcells.append({'seed':seed,'stage':i,'task':j,**counts,'delta_pp':float(delta*100),'AUC_contribution_pp':float(contribution*100),'R_group_steps':[r[g*10]['vector_group_steps'] for g in (0,1)],'C_group_steps':[c[g*10]['vector_group_steps'] for g in (0,1)]})
  task_contribution[j]=float(sum(ds)/len(ds)/4*100)
  rm=read(root/f'random/storage/task{j}.json');cm=read(root/f'coreset/storage/task{j}.json')
  rs={x['source']:x['sha256'] for x in rm['selected']};cs={x['source']:x['sha256'] for x in cm['selected']}
  assert len(rs)==len(cs)==rm['K']==cm['K'] and all(rs[x]==cs[x] for x in rs.keys()&cs.keys())
  storage.append({'seed':seed,'task':j,'K':rm['K'],'intersection':len(rs.keys()&cs.keys()),'different_members':len(rs.keys()-cs.keys()),'jaccard':len(rs.keys()&cs.keys())/len(rs.keys()|cs.keys()),'logged_C_radius':cm['selection']['covering_radius'][-1]})
 delta={k:arms['coreset']['metrics'][k]-arms['random']['metrics'][k] for k in ('AUC','FWT','NBT')}
 assert abs(sum(task_contribution.values())-delta['AUC'])<1e-10
 out['seeds'][str(seed)]={'arms':arms,'delta_pp':delta,'task_AUC_contribution_pp':task_contribution,'diagonal_AUC_contribution_pp':float(new_contribution*100),'post_learning_AUC_contribution_pp':float(old_contribution*100)}
 def diffs(a,b,path=''):
  if isinstance(a,dict) and isinstance(b,dict):
   return sum((diffs(a.get(k),b.get(k),path+'.'+k) for k in sorted(set(a)|set(b))),[])
  return [] if a==b else [{'path':path,'random':a,'coreset':b}]
 config_diffs[str(seed)]=diffs(read(root/'resolved_random.json'),read(root/'resolved_coreset.json'))
 assert {v['path'] for v in config_diffs[str(seed)]}=={'.buffer_dir','.experiment_dir','.run_root','.selector'}
 evaluation_diffs=diffs(read(root/'random/evaluation/resolved_actual.json'),read(root/'coreset/evaluation/resolved_actual.json'))
 assert {v['path'] for v in evaluation_diffs}=={'.experiment_dir'}
 out['seeds'][str(seed)]['actual_eval_config_differences']=evaluation_diffs
vals=[out['seeds'][str(s)]['delta_pp']['AUC'] for s in (42,43,44)]
out.update(cells=allcells,ifa=ifas,storage=storage,cost=cost,config_differences=config_diffs)
out['mean_task_AUC_contribution_pp']={str(t):statistics.mean(out['seeds'][str(s)]['task_AUC_contribution_pp'][t] for s in (42,43,44)) for t in range(6,10)}
out['AUC_delta_mean_pp']=statistics.mean(vals);out['AUC_delta_sample_sd_pp']=statistics.stdev(vals)
out['leave_one_seed_out_mean_pp']={str(s):statistics.mean(v for n,v in zip((42,43,44),vals) if n!=s) for s in (42,43,44)}
out['seed43_fraction_of_sum_delta']=vals[1]/sum(vals)
out['sign_randomization_two_sided_p']=sum(abs(sum(a*b for a,b in zip(signs,vals)))>=abs(sum(vals))-1e-10 for signs in itertools.product((-1,1),repeat=3))/8
assert all(sha(R/p)==h for p,h in manifest.items());out['input_hashes_after']='PASS'
(P/'RESULT_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ('cells','config_differences')},indent=2))
