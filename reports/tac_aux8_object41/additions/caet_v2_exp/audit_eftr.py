"""Independent ledger acceptance for native/candidate transactions."""
import json,math
from coreset_exp.common import O,sha
def js(p):return json.loads(p.read_text())
def audit():
    results={}
    assert js(O/'VARIANT_GPU_PREFLIGHT.json')['status']=='PASS'
    for t in range(6,10):
        accepted=rejected=transfer=trials=0
        if t>6:
            binding=js(O/'ra_coreset/effect_bindings'/('task%d.json'%t))
            assert binding['reference_stage']==t-1
            assert binding['reference_sha256']==sha(O/'ra_coreset/models'/('mlrifa_%d_model_ep99.pth'%(t-1)))
        for ep in range(100):
            c=js(O/'ra_coreset/constraints'/('task%d_epoch%02d.json'%(t,ep)))
            assert c['active']==(t>6)
            if t>6:assert c['samples']>0 and all(math.isfinite(v) for v in c['full_memory'].values())
            for x in js(O/'ra_coreset/auxiliary'/('task%d_epoch%02d.json'%(t,ep))):
                assert x['active']==(t>6)
                if t==6:assert x['candidate_optimizer_calls']==0;continue
                assert x['candidate_optimizer_calls']==2 and x['committed_updates']==1
                n=x['native_metrics'];a=x['candidate_metrics'];reasons=[]
                if a['bc'] is not None and a['bc']>n['bc']+.01:reasons.append('CURRENT_BC')
                if a['kl'] is not None:
                    if a['kl']>.05:reasons.append('EXEC_KL')
                    if a['switch']>.10:reasons.append('MODE_SWITCH')
                    if a['gripper']>.05:reasons.append('GRIPPER')
                assert reasons==x['rejection_reasons'] and x['candidate_accepted']==(not reasons)
                assert x['transfer_applied']==(not reasons and x['effect_loss'] is not None)
                accepted+=int(not reasons);rejected+=int(bool(reasons));transfer+=int(x['transfer_applied']);trials+=2
        results[str(t)]=dict(candidate_accepted=accepted,candidate_rejected=rejected,transfer_applied=transfer,optimizer_trial_calls=trials,committed_updates=(accepted+rejected) if t>6 else 15700)
    return results
