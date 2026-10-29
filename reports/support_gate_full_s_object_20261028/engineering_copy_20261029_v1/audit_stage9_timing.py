"""Read-only timing evidence for seed43 Task9; no training or environment work."""
from pathlib import Path
import datetime, hashlib, json, statistics, sys
D = Path(sys.argv[1])
v = json.loads((D / 'EVIDENCE.json').read_text(encoding='utf8'))
a = json.loads((D / 'LOCAL_AUDIT.json').read_text(encoding='utf8'))
assert a['status'] == 'PASS_PROGRESS_NOT_FINAL'
L = v['light']
get = lambda name: json.loads(L[name]['text'])
h = {s: get('seed%d/EPOCHS_S.json' % s) for s in [41, 42, 43]}
cap = get('engineering_copy_20261029_v1/SEED43_CAPTURE_EQ.json')
assert set(cap['checked']) == {'0', '15700', '38400', '69300'}
assert all(x['all_state_exact'] and x['optimizer_calls_added'] == 0 and x['environments_added'] == 0 for x in cap['checked'].values())
comparisons = {}
for stage in [7, 8, 9]:
    z = [r['seconds'] for r in h[43] if r['stage'] == stage]
    b = [r['seconds'] for r in h[42] if r['stage'] == stage][:len(z)]
    assert len(z) == len(b) and len(z) >= 10
    comparisons[str(stage)] = dict(matched_first_epochs=len(z), original_seed42_mean_seconds=statistics.mean(b), optimized_seed43_mean_seconds=statistics.mean(z), mean_time_reduction_fraction=1-statistics.mean(z)/statistics.mean(b), original_seed42_median_seconds=statistics.median(b), optimized_seed43_median_seconds=statistics.median(z), median_time_reduction_fraction=1-statistics.median(z)/statistics.median(b))
z = [r['seconds'] for r in h[43] if r['stage'] == 9]
assert 10 <= len(z) < 100 and a['counts']['43']['current']['stage'] == 9
assert all(sum(r['stage'] == t for r in h[43]) == 100 for t in [6, 7, 8])
assert a['formal_episodes'] == 0
obs = datetime.datetime.fromisoformat(a['observation'])
historical = D.parent / 'check_readonly_20261029_092535/TIMING_AUDIT.json'
old = json.loads(historical.read_text(encoding='utf8'))
out = dict(status='OBSERVED_STAGE9_PROGRESS_CONDITIONAL_ETA_NOT_FINAL', observation=a['observation'], snapshot_sha256=a['snapshot_sha256'], stage9_completed=len(z), recent10_mean_seconds=statistics.mean(z[-10:]), recent10_min_max_seconds=[min(z[-10:]), max(z[-10:])], all_stage9_mean_seconds=statistics.mean(z), remaining_compute_minutes_recent10=(100-len(z))*statistics.mean(z[-10:])/60, remaining_compute_minutes_all=(100-len(z))*statistics.mean(z)/60, cross_seed_matched_prefix_comparisons=comparisons, all_four_actual_stage_copy_checks=cap, planning_hours=dict(remaining_seed43_training_and_collection=[1,1.5], all_three_seed_formal=[1,3], total_training_and_formal=[2,4.5]), finish_window_training_and_formal=[(obs+datetime.timedelta(hours=x)).isoformat() for x in [2,4.5]], historical_600_formal_seconds=old['historical_600_formal_seconds'], historical_timing_audit_sha256=hashlib.sha256(historical.read_bytes()).hexdigest(), next_check_minutes=75, limitations=['Cross-seed comparisons differ in own cache content, data masks and resource conditions; they are not causal optimization speedup estimates.', 'Task9 is now directly measured. It is not assumed to share Task7/8 speedups.', 'Epoch timers exclude checkpoint saving, stage preparation and final ten training collection episodes.', 'Current formal evaluation has not begun; 1-3 hours is a historical planning allowance, not this run measurement.', 'Final independent CPU acceptance, analysis and publication are additional.', 'All scientific and engineering costs are unchanged; this audit performs no training or environment calls.'])
(D / 'TIMING_AUDIT.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
print(json.dumps(out, ensure_ascii=False, indent=2))
