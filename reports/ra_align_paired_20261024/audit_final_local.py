"""Independent scalar pair analysis. Never compares actions across different histories."""
import sys,json,hashlib,math,datetime
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]).resolve()
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
remote=js(D/'REMOTE_FINAL_AUDIT.json');assert remote['status']=='PASS_PRIVATE_HISTORY_ACTION_RECOMPUTATION' and remote['own_processes_absent'] and remote['source_cp_unchanged']
assert remote['new_forwards']==remote['new_environments']==remote['server_writes']==0
for n,v in remote['light_files'].items():assert sha(D/n)==v['sha256'] and (D/n).stat().st_size==v['bytes']
assert (D/'worker.exit').read_text().strip()==(D/'supervisor.exit').read_text().strip()=='0'
ad=js(D/'ADMISSION.json');assert len(ad['strict_checkpoints'])==12 and all(x['strict_keys']==100 for x in ad['strict_checkpoints'])
rec=js(D/'OUTCOMES.json');assert len(rec)==240
expected={(s,m,d,i) for s in [41,42,43] for m in ['RA','ALIGN'] for d in [8,9] for i in range(20)}
bykey={(x['seed'],x['method'],x['driver'],x['initial_index']):x for x in rec};assert set(bykey)==expected
rows_bykey={};total=0;trajectory=[];critical=[]
for key,r in bykey.items():
 assert r['attempt']==1 and r['same_history_reconstruction'] and 1<=r['policy_steps']<=600
 assert r['success'] or r['policy_steps']==600
 p=D/'light'/r['run']/('trajectory_%02d.jsonl'%r['initial_index']);assert sha(p)==r['rows_sha256']
 rows=[json.loads(x) for x in p.read_text().splitlines()];assert len(rows)==r['policy_steps'];rows_bykey[key]=rows
 for t,x in enumerate(rows):
  assert x['step']==t and x['history_length']==min(t+1,8) and x['same_history']
  assert x['mode_switch']==(x['teacher_mode']!=x['student_mode'])
  for k in ['mean_delta_squared','sigma_delta_squared','action_delta_squared']:assert len(x[k])==7 and all(math.isfinite(v) and v>=0 for v in x[k])
  assert math.isclose(sum(x['mean_delta_squared'])+sum(x['sigma_delta_squared']),x['w2_squared'],abs_tol=1e-10)
 total+=len(rows)
 trajectory.append({**r,'mode_switch_steps':sum(x['mode_switch'] for x in rows),'gripper_flip_steps':sum(x['gripper_flip'] for x in rows),'mean_w2_squared':float(np.mean([x['w2_squared'] for x in rows])),'p90_w2_squared':float(np.quantile([x['w2_squared'] for x in rows],.9))})
 indices=set(range(max(0,len(rows)-8),len(rows)))
 for t in r['first_events'].values():indices.update(range(max(0,t-4),min(len(rows),t+5)))
 critical.append({'key':list(key),'success':r['success'],'events':r['first_events'],'rows':[rows[t] for t in sorted(indices)]})
assert total==remote['steps']
pairs=[]
for seed in [41,42,43]:
 for method in ['RA','ALIGN']:
  for i in range(20):
   a=bykey[seed,method,8,i];b=bykey[seed,method,9,i];ra=rows_bykey[seed,method,8,i];rb=rows_bykey[seed,method,9,i]
   for n in ['initial_sha256','noise_sequence_sha256','env_seed']:assert a[n]==b[n]
   first_obs_mismatch=next((t for t,(x,y) in enumerate(zip(ra,rb)) if x['observation_sha256']!=y['observation_sha256']),None)
   same_start=ra[0]['observation_sha256']==rb[0]['observation_sha256']
   # Only same-history cross-driver points can corroborate shared decision differences.
   shared=[t for t,(x,y) in enumerate(zip(ra,rb)) if x['history_sha256']==y['history_sha256']]
   first_decision=next((t for t in shared if max(ra[t]['action_delta_squared'])>1e-12),None)
   pairs.append({'seed':seed,'method':method,'initial_index':i,'transition':('S' if a['success'] else 'F')+('S' if b['success'] else 'F'),'stage8_steps':a['policy_steps'],'stage9_steps':b['policy_steps'],'same_initial_observation':same_start,'first_observation_divergence_step':first_obs_mismatch,'shared_history_steps':len(shared),'first_action_difference_on_shared_history':first_decision,'stage8_history_first_events':a['first_events'],'stage9_history_first_events':b['first_events']})
summary=[]
for seed in [41,42,43]:
 for method in ['RA','ALIGN']:
  p=[x for x in pairs if x['seed']==seed and x['method']==method];cnt={k:sum(x['transition']==k for x in p) for k in ['SS','SF','FS','FF']}
  summary.append({'seed':seed,'method':method,**cnt,'stage8_success':cnt['SS']+cnt['SF'],'stage9_success':cnt['SS']+cnt['FS'],'delta_pp':5*(cnt['FS']-cnt['SF']),'all_same_initial_observation':all(x['same_initial_observation'] for x in p)})
for seed in [41,42,43]:
 # Also exact common-noise/initial cross-method pairing.
 for driver in [8,9]:
  for i in range(20):
   a=bykey[seed,'RA',driver,i];b=bykey[seed,'ALIGN',driver,i]
   assert a['initial_sha256']==b['initial_sha256'] and a['noise_sequence_sha256']==b['noise_sequence_sha256']
for name,value in [('PAIRED_OUTCOME_SUMMARY.json',summary),('PAIRED_TRAJECTORIES.json',pairs),('TRAJECTORY_SUMMARY.json',trajectory)]:
 (D/name).write_text(json.dumps(value,indent=2),encoding='utf8')
for seed in [41,42,43]:(D/('CRITICAL_SEGMENTS_seed%d.json'%seed)).write_text(json.dumps([x for x in critical if x['key'][0]==seed],separators=(',',':')),encoding='utf8')
audit={'status':'PASS_LOCAL_PAIRED_DIAGNOSTIC_AUDIT','episodes':240,'pairs':120,'vectors':24,'steps':total,'all_start_observations_equal':all(x['same_initial_observation'] for x in pairs),'source_cp_unchanged':True,'new_training_updates':0,'causal_protection_claim':False,'remote_evidence_sha256':sha(D/'REMOTE_FINAL_AUDIT.json'),'summary_sha256':sha(D/'PAIRED_OUTCOME_SUMMARY.json'),'notes':'Observed success-to-failure associations do not prove a single harmful decision or justify automatic training.'}
(D/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
lines=['# 配对轨迹诊断结果','','新的逐轨迹共同噪声协议；不是原global-RNG正式AUC重评。240回合、120 Stage8/9配对完整验收。','', '|seed|方法|Stage8成功/20|Stage9成功/20|SS|SF|FS|FF|变化pp|','|---|---|---|---|---|---|---|---|---|']
for x in summary:lines.append('|{seed}|{method}|{stage8_success}|{stage9_success}|{SS}|{SF}|{FS}|{FF}|{delta_pp:+d}|'.format(**x))
lines+=['','逐轨迹第一共同历史动作差异、首次观测分歧、同历史分量/夹爪事件和终止前标量段见PAIRED_TRAJECTORIES与CRITICAL_SEGMENTS。完整Temporal输入、动作与原始关键观测窗口保留服务器私有目录。原始RGB非全程保存。','', '尚须结合实际失败历史进行机制解释。超时不等于已知不可逆失败时刻；模式切换/夹爪翻转本身不证明有害。保护训练仍未启动，需依据可重复失败模式登记三组分叉及Task8保持/Task9学习协议。重复探索、有限seed/初态限制保留。']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(audit,ensure_ascii=False));print(json.dumps(summary,ensure_ascii=False))
