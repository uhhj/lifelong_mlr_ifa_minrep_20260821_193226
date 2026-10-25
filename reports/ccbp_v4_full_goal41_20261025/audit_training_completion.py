"""Local full-training metadata and capacity audit, not final CPU acceptance."""
import hashlib
import json
import sys
from pathlib import Path

snap,cap=map(Path,sys.argv[1:3])
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=read(snap/'EVIDENCE.json');L=v['light'];load=lambda n:json.loads(L[n]['text'])
for n,x in L.items():assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'],n
a=read(snap/'LOCAL_AUDIT.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['isolated_optimizer_calls']==66
assert a['training_collection_episodes']==120 and len(a['memory'])==12
minutes={};grad=0;valid=invalid=old=0;cache_counts={}
for arm in 'DOS':
    assert L['train_'+arm+'.exit']['text'].strip()=='0'
    tr=load('TRAIN_'+arm+'.json');h=load('EPOCHS_'+arm+'.json')
    assert tr['status']=='PASS' and tr['epochs']==len(h)==400 and tr['steps']==76100
    assert abs(sum(x['seconds'] for x in h)-tr['summed_epoch_seconds'])<1e-8
    assert tr['gradient_autograd_calls']==72 and tr['training_collection_episodes']==40
    minutes[arm]=tr['summed_epoch_seconds']/60;grad+=tr['gradient_autograd_calls']
    valid+=sum(x['totals']['valid'] for x in h);invalid+=sum(x['totals']['invalid'] for x in h);old+=sum(x['totals']['old_eligible'] for x in h)
    cache_counts[arm]={}
    for t in range(6,10):
        key='%s_%d'%(arm,t)
        assert L['collect_'+key+'.exit']['text'].strip()=='0'
        assert load('COLLECTION_STATE_PARITY_'+key+'.json')['parent_all_state_exact']
        m=load('MEMORY_'+key+'.json')
        assert tr['checkpoints'][str(t)]==load('STAGE_'+key+'.json')['sha256']==m['source_sha256']
        cache_counts[arm][str(t)]=m['count']
    if arm!='D':
        p=load('TASK6_PARITY_'+arm+'.json');assert p['same_device_D_equal']
        assert p['actual_sha256']==tr['checkpoints']['6']
        short=load('SHORT_HISTORY_ADMISSION_'+arm+'.json')
        assert short['status']=='PASS' and short['eight_frame_legacy_exact'] and short['all_state_restored']
        assert [r['length'] for r in short['lengths']]==list(range(1,9))
        assert all(r['finite_gradient'] and r['actual_slots']==4 for r in short['lengths'])
        assert short['extra_autograd_calls']==8 and short['policy_calls']==17
        assert short['optimizer_calls']==short['environments']==0
assert (valid,invalid,old)==(2143200,136200,811500)
out=dict(status='PASS_PROGRESS_NOT_FINAL',evidence_sha256=sha(snap/'EVIDENCE.json'),observation=v['observed'],
    epochs=1200,main_updates=228300,own_isolated_updates=66,formal_extra_autograd=grad,
    short_history_extra_autograd=16,short_history_policy_calls=34,training_collection_episodes=120,
    BC_valid=valid,BC_masked=invalid,old_valid=old,epoch_minutes=minutes,total_epoch_minutes=sum(minutes.values()),
    cache_counts=cache_counts,parent_state_parity_all12=True,OS_Task6_tensor_equal_D_server_report_bound=True,
    final_CPU_cache_replay_initial_bytes_and_scientific_acceptance='PENDING')
(snap/'TRAINING_COMPLETION_METADATA_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
c=read(cap/'EVIDENCE.json');assert c['status']=='PASS_NOT_FINAL'
assert all(c[k]==0 for k in ['payload_reads','forwards','updates','environments','server_writes'])
for name,x in c['volumes'].items():
    assert x['headroom']==x['free']-x['remaining_bound']-x['reserve']>=0
    assert x['reserve']==(2 if name=='/data' else 1)*1024**3
(cap/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',evidence_sha256=sha(cap/'EVIDENCE.json'),observation=c['observed'],volumes=c['volumes'],no_payload_or_remote_mutation=True),indent=2)+'\n',encoding='utf8')
print(json.dumps(out,indent=2))
