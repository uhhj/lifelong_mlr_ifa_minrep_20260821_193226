from pathlib import Path
import json,subprocess,sys,hashlib
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'))
assert v['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and not v['processes']
subprocess.run([sys.executable,'-X','utf8',str(R/'audit_progress.py'),str(D)],check=True)
a=json.loads((D/'LOCAL_AUDIT.json').read_text(encoding='utf8'));assert not a['failures'] and a['epochs']==1700 and a['accepted_updates']==403000 and a['physical_training_optimizer_calls']==494700 and a['formal_episodes']==1120 and a['own_isolated_optimizer_calls']==493
assert v['light']['controller.exit']['text']=='0'
models=0;cases=json.loads((R/'CASES.json').read_text(encoding='utf8'))
for name,c in cases.items():
    x=v['cases'][name];assert x['status']=='PASS_REMOTE_CPU_PENDING_LOCAL' and len(x['strict'])==len(c['arms']);models+=len(x['strict'])
    assert x['new_optimizer_updates']==x['new_environments']==0
    def js(n):return json.loads(v['light'][name+'/'+n]['text'])
    records=js('OUTCOMES.json');shards=[]
    for n,l in v['light'].items():
        if n.startswith(name+'/outcomes/'):shards.extend(json.loads(l['text']))
    key=lambda r:(r['arm'],r['task'],r['initial_index'])
    assert sorted(records,key=key)==sorted(shards,key=key)
    events=[json.loads(s) for s in v['light'][name+'/EVAL_EVENTS.jsonl']['text'].splitlines()]
    assert len(events)==len(records)//5
    for i in range(0,len(events),2):assert events[i]['kind']=='start' and events[i+1]['kind']=='complete' and events[i]['run']==events[i+1]['run']
    for arm in c['arms']:
        strict=x['strict'][arm];assert strict['strict_keys']==100 and strict['complete_recovery'] and strict['common_initial_exact']
        assert js('TRAIN_'+arm+'.json')['sha256']==strict['sha256']
    for arm in c['arms']:
        # Same suite/seed/task initial bytes and trajectory noise across methods.
        ref={key(r)[1:]:r for r in records if r['arm']=='C'}
        for r in [r for r in records if r['arm']==arm]:
            old=ref[key(r)[1:]];assert all(r[k]==old[k] for k in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed'])
for n,x in v['diagnostic_files'].items():
    p=D/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'];rows=json.loads(p.read_text())
    for row in rows:
        c=row['component_costs'];z=row['logits'];selected=max(range(len(z)),key=lambda k:z[k]);assert selected==row['selected'] and abs(c[selected]-row['executed_cost'])<1e-12 and abs(c[selected]-min(c)-row['regret'])<1e-12
assert models==17
out=dict(status='PASS_COMPLETE_LOCAL_ACCEPTANCE',strict_CPU_checkpoints=models,epochs=1700,accepted_updates=403000,physical_training_optimizer_calls=494700,isolated_optimizer_calls=493,formal_episodes=1120,source_evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),mechanism_only=True,full_OBJECT_S_followup_still_required=True)
(D/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out))
