import json,hashlib,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
inputs=[P/'DIAGNOSTICS.json',P.parent/'object_five_seed_readonly_analysis/ANALYSIS.json']
hashes={p.relative_to(P.parent).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
expected=json.loads((P/'INPUT_SHA256.json').read_text(encoding='utf-8'))
assert hashes==expected, 'Published input hash mismatch'
d=json.loads(inputs[0].read_text(encoding='utf-8'));old=json.loads(inputs[1].read_text(encoding='utf-8'))
out={'status':'PASS_READONLY','input_sha256':hashes,'tasks':{},'sorted_slots':d['sorted_slot_analysis'],'epoch_logs':d['epoch_log_comparison']}
for task in range(6,10):
    rs=[r for r in d['rows'] if r['task']==task]
    added=sum(r['added'] for r in rs)
    def weighted(key,group):
        return sum(r['action_stats'][group].get(key,0)*r[group if group=='added' else 'removed'] for r in rs)/added if added else None
    out['tasks'][task]={'added_across_seed_selections':added,'newly_covered_events':sum(r['newly_covered_events'] for r in rs),
        'added_with_already_covered_event':sum(r['added_windows_with_previously_covered_event'] for r in rs),
        'c_covered_events_per_seed':[r['c_covered_events'] for r in rs],
        'ec_covered_events_per_seed':[r['ec_covered_events'] for r in rs],
        'confirmed_events_per_seed':[r['confirmed_events'] for r in rs],
        'removed_within7_of_event_boundary':sum(r['near_event_boundary']['removed']['within7_either'] for r in rs),
        'added_close_fraction':weighted('gripper_close_fraction','added'),'removed_close_fraction':weighted('gripper_close_fraction','removed'),
        'added_translation_rms':weighted('translation_rms_mean','added'),'removed_translation_rms':weighted('translation_rms_mean','removed'),
        'added_rotation_rms':weighted('rotation_rms_mean','added'),'removed_rotation_rms':weighted('rotation_rms_mean','removed'),
        'adjacent_target_pairs_increase':[r['positions']['EC']['adjacent_target_pairs']-r['positions']['C']['adjacent_target_pairs'] for r in rs],
        'missed_with_eligible_window':[[e for e in r['per_event'] if e['candidate_windows']>0 and e['c_windows']==0] for r in rs]}
assert all(hashlib.sha256((P.parent/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
(P/'SUMMARY.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'status':out['status'],'task7_added':out['tasks'][7]['added_across_seed_selections'],'task8_added':out['tasks'][8]['added_across_seed_selections'],'task7_8_new_event_coverage':out['tasks'][7]['newly_covered_events']+out['tasks'][8]['newly_covered_events']},indent=2))
