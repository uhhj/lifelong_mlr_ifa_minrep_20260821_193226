"""Actual full acceptance, scalar recomputation and all paired cells. No forwards."""
import json,sys,hashlib,subprocess
from pathlib import Path
from fractions import Fraction as F
import numpy as np
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=js(D/'EVIDENCE.json');assert v['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and v['no_residual'] and v['environment_calls']==v['optimizer_updates']==0
assert v['initial_state_bytes_verified'] and v['reference_C_pairing_exact'] and v['task6_current_arms_exact']
subprocess.run([sys.executable,str(R/'audit_progress.py'),str(D)],check=True);a=js(D/'LOCAL_AUDIT.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and all(x['epochs']==400 and x['steps']==76100 for x in a['counts'].values())
assert a['isolated_optimizer_calls']==66 and a['formal_episodes']==600 and a['training_collection_episodes']==120
subprocess.run([sys.executable,str(R/'audit_evaluation_progress.py'),str(D)],check=True);e=js(D/'EVALUATION_PROGRESS_AUDIT.json')
assert e['completed_vectors']==60 and e['closed_cells']==30 and not e['ongoing']
L={n:json.loads(x['text']) if n.endswith('.json') else x['text'] for n,x in v['light'].items()};assert L['controller.exit']=='0' and L['evaluate.exit']=='0'
events=[json.loads(x) for x in L['COLLECTION_EVENTS.jsonl'].splitlines()];starts={};ends={}
for x in events:
 key=(x['arm'],x['task']);target=starts if x['kind']=='start' else ends;assert key not in target and x['kind'] in ['start','complete'];target[key]=x
assert set(starts)==set(ends)=={(arm,t) for arm in 'DOS' for t in range(6,10)}
for key,x in ends.items():assert x['time']>=starts[key]['time'] and x['attempts']==10
collection_cost={};gradients={}
for arm in 'DOS':
 assert L['train_'+arm+'.exit']=='0' and L['TRAIN_'+arm+'.json']['gradient_autograd_calls']==72
 gg=L['GRADIENTS_'+arm+'.json'];assert {(x['stage'],x['epoch']) for x in gg}=={(t,ep) for t in range(7,10) for ep in [0,9,49,99]}
 assert sum(x['extra_autograd_calls'] for x in gg)==72
 for x in gg:
  for k in ['bc','ifa_weighted','expert_weighted','expert_route_weighted','retain_weighted','retain_route_weighted']:assert np.isfinite([x[k]['loss'],x[k]['l2']]).all() and x[k]['l2']>=0
 gradients[arm]=gg
 assert all(v['strict'][arm][str(t)]['strict_keys']==100 for t in range(6,10)) and v['strict'][arm]['recovery']['final_hardlink']
 collection_cost[arm]={}
 for t in range(6,10):
  key=arm+'_'+str(t);m=L['MEMORY_'+key+'.json'];vm=v['memory_audits'][key]
  assert vm['cache_sha256']==m['cache_sha256'] and vm['source_sha256']==v['strict'][arm][str(t)]['sha256'] and vm['count']==m['count']
  assert vm['sampling_exact'] and vm['scale_exact'] and L['COLLECTION_STATE_PARITY_'+key+'.json']['parent_all_state_exact']
  collection_cost[arm][str(t)]={k:m[k] for k in ['count','collection_successes','training_attempts','online_frontend_calls','online_policy_calls','demonstration_policy_calls','parity_frontend_calls','parity_policy_calls','seconds','terminal_task_unused_for_future_training']}
records=L['OUTCOMES.json']+js(R/'REFERENCE_C.json')['records'];matrix={};metrics={};by={}
def metric(r):
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
for arm in 'CDOS':
 rr={}
 for stage in range(6,10):
  for task in range(6,stage+1):
   rows=sorted([x for x in records if (x['arm'],x['stage'],x['task'])==(arm,stage,task)],key=lambda x:x['initial_index']);assert [x['initial_index'] for x in rows]==list(range(20))
   by[arm,stage,task]=rows;rr[stage,task]=F(sum(x['success'] for x in rows),20)
 matrix[arm]={'%d_%d'%k:dict(successes=int(x*20),percent=float(x*100),episodes=20) for k,x in rr.items()}
 metrics[arm]={k:dict(fraction=str(x),percent=float(x*100)) for k,x in metric(rr).items()}
pairs={};negative=[]
for aa,bb in [('D','C'),('O','C'),('S','C'),('O','D'),('S','D'),('S','O')]:
 cells={}
 for stage in range(6,10):
  for task in range(6,stage+1):
   ca=by[aa,stage,task];cb=by[bb,stage,task];assert all(all(x[k]==y[k] for k in ['initial_sha256','initial_file_sha256','noise_sha256','env_seed']) for x,y in zip(ca,cb))
   sf=[x['initial_index'] for x,y in zip(ca,cb) if y['success'] and not x['success']];fs=[x['initial_index'] for x,y in zip(ca,cb) if not y['success'] and x['success']];delta=5*(len(fs)-len(sf))
   cells['%d_%d'%(stage,task)]=dict(delta_pp=delta,reference_success_to_candidate_failure=sf,reference_failure_to_candidate_success=fs)
   if delta<0:negative.append(dict(comparison=aa+'-'+bb,stage=stage,task=task,delta_pp=delta))
 pairs[aa+'-'+bb]=cells
def supported(aa,bb):return metrics[aa]['OldFinal']['percent']>metrics[bb]['OldFinal']['percent'] and all(matrix[aa][k]['successes']>=matrix[bb][k]['successes'] for k in ['9_8','9_9','7_6'])
support={k:supported(*k.split('-')) for k in ['D-C','O-C','S-C','O-D','S-O']}
summaries={};identities={};total_expert=0;total_retain=0
for name,meta in v['diagnostic_files'].items():
 p=D/name;assert sha(p)==meta['sha256'];rows=js(p);assert len(rows)==meta['rows'];kind,arm,st=p.stem.split('_');stage=int(st);assert kind in ['expert','retain'] and arm in 'DOS'
 if kind=='expert':
  assert len(rows)=={7:477,8:899,9:1329}[stage];identities[arm,stage]=[(r['index'],r['task'],r['sample_sha256']) for r in rows];total_expert+=len(rows)
 else:
  assert len(rows)==sum(L['MEMORY_%s_%d.json'%(arm,t)]['count'] for t in range(6,stage));total_retain+=len(rows)
 scored=[]
 for row in rows:
  groups=np.asarray(row['component_group_costs' if kind=='expert' else 'component_group_distances'],dtype=float);assert groups.shape==(5,3) and np.isfinite(groups).all() and np.all(groups>=0)
  cc=groups.sum(-1);z=np.asarray(row['logits']);assert z.shape==(5,) and np.isfinite(z).all();k=int(z.argmax());best=int(cc.argmin());assert k==row['student_mode']
  route=max(0.,.1+float(np.max(np.delete(z,best)))-float(z[best]))
  scored.append(dict(index=row['index'],task=row['task'],execution_cost=float(cc[k]),best_cost=float(cc.min()),regret=float(cc[k]-cc.min()),margin_route=route,auxiliary=float(cc[k])+.1*route))
 stats={}
 for t in sorted({r['task'] for r in scored}):
  rr=[r for r in scored if r['task']==t];stats[str(t)]=dict(count=len(rr),nonbest_count=sum(r['regret']>1e-12 for r in rr),statistics={k:dict(mean=float(np.mean([r[k] for r in rr])),median=float(np.median([r[k] for r in rr])),p90=float(np.percentile([r[k] for r in rr],90)),maximum=float(max(r[k] for r in rr))) for k in ['execution_cost','best_cost','regret','margin_route']})
 summaries[p.stem]=stats;(D/('RECOMPUTED_'+p.stem+'.json')).write_text(json.dumps(dict(rows=scored,by_task=stats),indent=2)+'\n',encoding='utf8')
assert len(summaries)==18 and total_expert==8115
for stage in range(7,10):assert identities['D',stage]==identities['O',stage]==identities['S',stage]
cache_windows=sum(x['count'] for x in v['memory_audits'].values())
assert v['diagnostic_windows']==total_expert+total_retain+cache_windows and v['diagnostic_calls']==258+total_retain+cache_windows
out=dict(status='PASS_COMPLETE_LOCAL_ACCEPTANCE',contract=v['contract_sha256'],scientific_exploratory_support=support,no_automatic_expansion=True,matrix=matrix,metrics=metrics,paired=pairs,all_negative_cells=negative,offline_by_stage_task=summaries,training=dict(epochs=1200,updates=228300,own_isolated_optimizer_updates=66,formal_extra_autograd_calls=216,admission_extra_autograd_calls=54,epoch_seconds={arm:L['TRAIN_'+arm+'.json']['summed_epoch_seconds'] for arm in 'DOS'},training_collection_episodes=120,collection_cost=collection_cost),evaluation=dict(episodes=600,vectors=60,cells=30,reference_C_episodes_reused=200),final_CPU=dict(strict_checkpoints=12,complete_recoveries=3,cache_files=12,calls=v['diagnostic_calls'],windows=v['diagnostic_windows'],seconds=v['diagnostic_seconds']),evidence_sha256=sha(D/'EVIDENCE.json'),limits=['Single repeatedly developed seed and evaluation initial states; not confirmation or significance.','Existing C had no additional training collection; DOS matched attempt budgets, not equal runtime or retained sample counts.','S filters whole trajectories, not step optimality; empty tasks contribute zero and may lose protection.','Own-policy histories and teachers change across arms; O-S is the full filtering intervention, not isolated label quality.','Offline retention costs use different arm-specific histories; common expert windows are checked separately.','Online cached feature hashes and source-policy replay are verified; no raw-camera/physics replay or causal failure timing is claimed.'])
(D/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
lines=['# CCBP-v4 GOAL seed41 D/O/S最终验收','', '独立验收完成；探索支持条件：'+str(support)+'。不自动扩展。','','|组别|AUC|FWT|NBT（低好）|OldFinal|FinalAll|','|---|---:|---:|---:|---:|---:|']
for arm in 'CDOS':lines.append('|'+arm+'|'+'|'.join('%.4f'%metrics[arm][k]['percent'] for k in ['AUC','FWT','NBT','OldFinal','FinalAll'])+'|')
lines+=['','|阶段/任务|C|D|O|S|','|---|---:|---:|---:|---:|']
for stage in range(6,10):
 for task in range(6,stage+1):lines.append('|%d/%d|'%(stage,task)+'|'.join(str(matrix[arm]['%d_%d'%(stage,task)]['percent']) for arm in 'CDOS')+'|')
lines+=['','全部配对SF/FS和负单元见LOCAL_FINAL_AUDIT。离线保持成本因缓存来源不同不能直接跨组当同分布误差；共同专家窗口的误差另列。120训练采集与600正式环境分开计费/计量。C只读复用，非交互等资源。','',*out['limits']]
(D/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],support=support,metrics=metrics),indent=2))
