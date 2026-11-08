"""Recompute remaining evaluation time from actual complete vector events."""
from pathlib import Path
import datetime,hashlib,json,statistics,sys
D=Path(sys.argv[1]);read=lambda p:json.loads(p.read_text(encoding='utf8'))
v=read(D/'EVIDENCE.json');a=read(D/'LOCAL_AUDIT.json');L=v['light']
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['epochs']==1200
stats={}
for case in ('Q1_41','Q1_42','Q1_43'):
 assert L[case+'/train_S.exit']['text'].strip()=='0'
 events=[json.loads(x) for x in L[case+'/EVAL_EVENTS.jsonl']['text'].splitlines()]
 done=[e for e in events if e['kind']=='complete'];times=[e['seconds'] for e in done]
 assert times
 stats[case]=dict(complete_vectors=len(done),outcome_episodes=a['counts'][case]['formal_episodes'],
     mean_seconds=statistics.mean(times),recent5_seconds=statistics.mean(times[-5:]),
     remaining_vectors=20-len(done),linear_remaining_minutes=(20-len(done))*statistics.mean(times)/60)
obs=datetime.datetime.fromisoformat(a['observation']);remain=sum(x['remaining_vectors'] for x in stats.values())
assert remain==10, 'Reassess planning interval for changed progress'
controller=[json.loads(x) for x in L['CONTROLLER_EVENTS.jsonl']['text'].splitlines()]
exits=[dict(case=e['case'],time=datetime.datetime.fromtimestamp(e['time'],obs.tzinfo).isoformat(),returncode=e['returncode']) for e in controller if e['kind']=='train_exit']
out=dict(status='ACTUAL_EVALUATION_IN_PROGRESS_NOT_FINAL_ACCEPTANCE',observed=a['observation'],
    training_exits=exits,evaluation=stats,remaining_episodes=600-a['formal_episodes'],
    planning_remaining_minutes=[10,20],planning_completion=[(obs+datetime.timedelta(minutes=m)).isoformat() for m in (10,20)],
    next_check_minutes=12,source_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),
    limits=['Task lengths and success termination vary; current vector is not treated as complete.',
            'Controller has not exited. Final CPU acceptance must await actual exit0 and no own processes.',
            'CPU acceptance, interpretation and publication are additional.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
