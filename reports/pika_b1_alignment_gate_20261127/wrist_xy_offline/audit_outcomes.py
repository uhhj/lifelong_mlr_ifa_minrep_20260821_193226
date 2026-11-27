"""Recompute the published headline values from lightweight per-plan and episode audits."""
import json
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parent


def verify():
    outcomes=json.loads((ROOT/'outcomes.json').read_bytes())
    gate=json.loads((ROOT/'guard_cases.json').read_bytes())
    axes=json.loads((ROOT/'axis_check.json').read_bytes())
    assert outcomes['sha256']!=outcomes['active_station']['sha256']
    assert outcomes['duration']==outcomes['active_station']['duration'] and outcomes['end_mode']=='stop'
    summary={}
    assert len(outcomes['field'])==2
    for trial in outcomes['field']:
        heads=trial['fixed_heads']
        old=median([x['old_mm'] for x in heads])
        new=median([x['new_mm'] for x in heads])
        assert abs(trial['baseline_signed_mm']['median']-old)<=.00001
        assert abs(trial['candidate_signed_mm']['median']-new)<=.00001
        assert all(h['axis_cos']>.99 for h in heads)
        assert trial['original_reconstruction_max_mm']<.001
        for mode in ('schedule','guarded'):
            replay=trial[mode]
            assert replay['frames']==trial['recorded_frames'] and not replay['grip_targets_mm']
            assert replay['max_z_change_mm']<.001 and replay['max_width_change_mm']==0 and replay['offsets']['7']>0
        summary[trial['reference_axis']]=(len(heads),round(old,3),round(new,3))
    assert summary['bottom'][2]>1 and abs(summary['right'][2])<.02
    assert not outcomes['expert_close_differences']
    for name,episodes in outcomes['controls']['episodes'].items():
        c=outcomes['controls']['counts'][name]
        b=outcomes['baseline_expert_counts'][name]
        assert len(episodes)==c['windows']==b['windows']
        assert c['close_accepted']==b['close_accepted']==sum(r['close_accepted'] for r in episodes)
        assert all(r['raw_side_at_expert_close'] is None for r in episodes)
        assert sum(r['center_cap'] for r in episodes)==c['center_cap']
    assert len(gate)==15 and {r['case'] for r in gate}>={'deadline','heartbeat','stale_prediction','no_stop_ack'}
    for k,v in axes.items():
        assert v['informative']==sum(p['informative'] for p in v['pairs'])
        assert v['agree']==sum(p['agrees'] and p['informative'] for p in v['pairs'])
    return summary


if __name__=='__main__':
    print(json.dumps(verify(),ensure_ascii=False))
