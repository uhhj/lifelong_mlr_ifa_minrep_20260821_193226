"""Read four archived runs and quantify width before the first USB width target."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys


def extract(folder):
    sessions=('212732_689056','213025_690417','213755_694001','223635_725397')
    summary=[]
    for session in sessions:
        path=folder/f'20261127_{session}_B1-ode50-aligned-20261125.jsonl'
        raw=path.read_bytes()
        rows=[json.loads(line) for line in raw.splitlines()]
        sent=[r for r in rows if r.get('mode')=='execute' and r.get('sent')]
        start=next(r for r in rows if r['event']=='start')
        stable=next(r for r in rows if r['event']=='gripper_stable')
        trace=next(r for r in rows if r['event']=='pika_stop_trace')
        end=rows[-1]
        goals=[r for r in sent if r.get('pika_target_mm') is not None]
        stop=trace['stop_requested_ns']
        first=goals[0]['send_started_ns'] if goals else stop
        widths=[r['measured_at_send'][6] for r in sent if r['send_started_ns']<first]
        base=start['initial'][6]
        item=dict(session=path.stem,source_sha256=hashlib.sha256(raw).hexdigest(),dispatches=len(sent),
                  enable_reference_observation_mm=round(base,3),stable_mm=round(stable['initial'][6],3),
                  before_width_target_or_stop_s=round((first-stable['mono_ns'])/1e9,3),
                  dispatch_widths_before_first_target_mm=[round(min(widths),3),round(max(widths),3)],
                  width_goals_mm=[r['pika_target_mm'] for r in goals])
        summary.append(item)
        if session != sessions[-1]:continue
        lo,hi=trace['bounds_mm']
        frames=trace['frames']
        pre=[f for f in frames if f['rx_ns']<stop]
        bad=[f for f in pre if f['width_mm'] is not None and not lo<=f['width_mm']<=hi]
        assert len(bad)==1
        assert not goals and len(sent)==136 and end['event']=='end'
        assert any(r['event']=='start_confirmed' for r in rows) and rows[0]['end_mode']=='stop'
        assert not any(r['event'] in ('send_interrupted','target_guard_fault') for r in rows)
        def snapshot(f):
            motor=json.loads(f['usb_frame'])['motor']
            return dict(ms_from_stop=round((f['rx_ns']-stop)/1e6,2),width_mm=round(f['width_mm'],4),
                        voltage_v=f['voltage_v'],current_ma=f['current_ma'],status=f['status'],
                        motor_position=motor['Position'],motor_speed=motor['Speed'])
        centered=[r for r in sent if r['grasp_alignment']['aligned']]
        overhead=[r for r in sent if r['grasp_alignment']['distance_mm']<=20.]
        inside=[r for r in overhead if r['grasp_alignment']['wrist_box'] is not None]
        prevolt=[f['voltage_v'] for f in pre if f.get('voltage_v') is not None]
        last=sent[-1]
        item.update(end_detail=end['detail'],outcome=end['outcome'],bounds_mm=[lo,hi],
                    reference_midpoint_mm=(lo+hi)/2,first_outside=snapshot(bad[0]),
                    missed_lower_by_mm=round(lo-bad[0]['width_mm'],5),
                    last_40_ms_before_stop=[snapshot(f) for f in pre if f['rx_ns']>=stop-40_000_000],
                    first_after_stop=snapshot(next(f for f in frames if f['rx_ns']>=stop)),
                    first_disabled=snapshot(next(f for f in frames if f['rx_ns']>=stop and f['status']==0)),
                    pre_stop_window_voltage_v=[min(prevolt),max(prevolt)],
                    pre_stop_window_status_counts=dict(Counter(f['status'] for f in pre)),
                    phase_counts=dict(Counter(r['gripper_phase'] for r in sent)),
                    arm_correction_phase_counts=dict(Counter(r.get('grip_plan_phase') for r in sent)),
                    candidates_with_close_intent=sum(r['close_intent'] for r in sent),
                    overhead_close_frames=len(overhead),box_in_overhead_close_frames=len(inside),aligned_frames=len(centered),
                    first_overhead_close_s=round((overhead[0]['stamp']-stable['mono_ns'])/1e9,3) if overhead else None,
                    final_overhead_distance_mm=round(last['grasp_alignment']['distance_mm'],2),
                    last_dispatch_to_stop_ms=round((stop-last['send_finished_ns'])/1e6,2))
    return summary


if __name__=='__main__':
    print(json.dumps(extract(Path(sys.argv[1])),ensure_ascii=False,indent=2))
