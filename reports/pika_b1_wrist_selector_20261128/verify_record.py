"""Verify the episode summary, published planning statistics and installed version identities."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

root=Path(__file__).resolve().parent
def data(n):return json.loads((root/n).read_bytes())
def verify(raw=None):
    d=data('outcomes.json');installed=data('installation.json');tests=data('deployment_tests.json')
    plans=[x for x in d['plan_evidence'] if x['s']>=4.]
    assert len(plans)==8 and sum(x['baseline_norm_mm']==0 for x in plans)==5
    assert sum(x['baseline_signed_mm']<0 for x in plans)==3
    assert all(x['candidate_signed_mm']>0 for x in plans)
    assert abs(median(x['candidate_signed_mm'] for x in plans)-d['late_plan_stats']['candidate_signed_mm']['median'])<1e-5
    assert installed['source_bundle']==d['actual_profile'] and installed['new_bundle']==d['candidate_profile']
    assert d['desktop_inventory']['candidate_keys_in_active']==[]  # observed before installation
    for key,v in installed['installed_replay'].items():
        assert v['frames']==d['frames']==188 and not v['gripper_events'] and v['device_writes']==0
        checks=tests['selection_and_guards'][key]['cases']
        assert len(checks)==15 and {x['case'] for x in checks}>={'deadline','confirm_only','heartbeat','no_stop_ack'}
    if raw is not None:
        log=raw/(d['session']+'.jsonl');images=raw/'inputs.jsonl'
        for p,k in [(log,'log_sha256'),(images,'inputs_sha256')]:assert hashlib.sha256(p.read_bytes()).hexdigest()==d[k]
        rows=[json.loads(l) for l in log.read_bytes().splitlines()]
        sent=[r for r in rows if r.get('sent') and r.get('mode')=='execute']
        assert d['actual_profile']==rows[0]['policy']['sha256']
        assert len(sent)==d['frames'] and Counter(r['gripper_phase'] for r in sent)==d['phase_counts']
        assert sum(r['close_intent'] for r in sent)==d['intent']
        assert sum(r['grasp_alignment']['aligned'] for r in sent)==d['passed']==0
        assert [r['pika_target_mm'] for r in sent if r['pika_target_mm'] is not None]==d['goals_mm']==[]
        assert rows[-1]['detail']==d['end']
    print(f"Verified {len(plans)} late plans, {d['frames']} commands and {installed['installed_guard_cases']} installed guard cases.")

if __name__=='__main__':
    verify(Path(sys.argv[1]) if len(sys.argv)>1 else None)
