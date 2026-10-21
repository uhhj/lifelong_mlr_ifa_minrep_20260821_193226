"""Read-only admission acceptance before the first complete training epoch."""
import argparse
import datetime
import json
from pathlib import Path
from audit_caet_progress import R, read, sha

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    ap.add_argument('--seed', type=int, choices=(41, 42, 43), required=True)
    args = ap.parse_args()
    p = args.directory.resolve()
    assert p.parent == R
    name = 'CAET_DETAILS.json' if args.seed == 42 else f'CAET_DETAILS_seed{args.seed}.json'
    d = read(p/name); c = d['caet']; s = read(p/'SNAPSHOT.json')
    assert d['status'] == 'CAET_FROZEN_CONTRACT_INPUTS_PASS'
    assert c['source_input_feature_hashes_pass'] and c['source_files'] == 204
    assert c['launch']['status'] == c['preflight']['status'] == 'PASS'
    assert c['launch']['contract_sha256'] == c['contract_sha256']
    assert c['preflight']['optimizer_calls'] == 4 and c['preflight']['environment_steps'] == 0
    assert all(v == '0' for v in c['admission_exits'].values())
    # Startup snapshots do not distinguish an absent INITIAL_STATE file from
    # a mismatched one. Do not certify parity from this early observation.
    assert all(c['ifa_equal'].values())
    assert c['status']['phase'] == 'train6' and c['status']['status'] == 'RUNNING'
    assert c['epochs'] in (None, []), 'Use full progress audit after complete epoch metadata exists'
    assert any('-m caet_exp.train ' in w['command'] for w in c['workers'])
    assert s['controller_v2']['exit'] is None
    assert (s['decision']['method'], s['decision']['suite'], s['decision']['seed']) == ('CAET','OBJECT',args.seed)
    assert len(c['formal']['trials']) == 0
    out = dict(status='PARTIAL', audit='CAET_ADMISSION_AND_LIVE_TRAINING_PASS', seed=args.seed,
        observed_at=datetime.datetime.fromtimestamp(d['observed_end_unix'], datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        phase='train6', completed_epoch_metadata_available=False, source_files=204,
        preflight=c['preflight'], admission_exits=c['admission_exits'], workers=c['workers'],
        details_sha256=sha(p/name), snapshot_sha256=sha(p/'SNAPSHOT.json'),
        initial_rng_parity='REQUIRES_LATER_PROGRESS_AUDIT',
        initial_rng_equal_observed=c['initial_rng_equal'],
        task6_final_tensor_parity='PENDING_TASK6_COMPLETE', formal_records=0,
        optimizer_calls_this_audit=0, environment_steps_this_audit=0)
    (p/f'CAET_LAUNCH_AUDIT_seed{args.seed}.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__ == '__main__':
    main()
