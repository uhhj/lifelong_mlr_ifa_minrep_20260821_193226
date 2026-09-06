"""Inspect the prespecified 5 losses and 11 gains using existing traces only."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/paired_rollout_diagnosis'
OUT = ROOT / 'reports/paired_rollout_diagnosis'


def main():
    rows = list(csv.DictReader((OUT / 'paired_trials.csv').open()))
    selected = [r for r in rows if (
        r['case'] == 'seed42_stage9_task7' and r['control'] == 'mlr'
        and r['corrected_success'] == 'False' and r['control_success'] == 'True') or (
        r['case'] == 'seed43_stage9_task9' and r['control'] == 'release_ifa'
        and r['corrected_success'] == 'True' and r['control_success'] == 'False')]
    selected.sort(key=lambda r: (r['case'], int(r['trial'])))
    assert [int(r['trial']) for r in selected if r['case'] == 'seed42_stage9_task7'] == [5,8,16,17,18]
    assert [int(r['trial']) for r in selected if r['case'] == 'seed43_stage9_task9'] == [4,6,9,11,12,13,14,16,17,18,19]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    records, manifest = [], []
    for row in selected:
        case, trial = row['case'], int(row['trial'])
        round_id, lane = divmod(trial,10)
        object_name = 'milk_1' if case.endswith('task7') else 'orange_juice_1'
        canvas = Image.new('RGB', (7*150, 4*165+90), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((6,5), f'{case} trial={trial}; recorded frames every50 steps; no new simulation', fill='black', font=font)
        for arm_index, arm in enumerate((row['control'], 'corrected_ifa')):
            path = RUN / case / arm
            info = json.loads((path / f'round{round_id}.json').read_text())[lane]
            end = info['first_success_step'] or info['policy_steps']
            with np.load(path / f'round{round_id}.npz') as trace:
                keep = (trace['observation_steps'] >= 0) & (trace['observation_steps'] <= end)
                steps = trace['observation_steps'][keep]
                obj = trace[f'obs__{object_name}_pos'][keep,lane]
                eef = trace['obs__robot0_eef_pos'][keep,lane]
                basket = trace['obs__basket_1_pos'][keep,lane]
                rise = obj[:,2]-obj[0,2]
                xy = np.linalg.norm(obj[:,:2]-basket[:,:2],axis=1)
                gap = np.linalg.norm(obj-eef,axis=1)
                eef_basket_xy = np.linalg.norm(eef[:,:2]-basket[:,:2],axis=1)
                unsupported_near = np.flatnonzero((eef_basket_xy <= .10) & (xy > .10))
                lifted = np.flatnonzero(rise >= .03)
                peak = int(np.argmax(rise))
                down = np.flatnonzero((np.arange(len(rise)) > peak) & (rise < .03)) if len(lifted) else []
                record = dict(case=case, trial=trial, arm=arm, success=info['success'],
                    first_success_step=info['first_success_step'],
                    first_3cm_rise_step=int(steps[lifted[0]]) if len(lifted) else None,
                    max_rise_m=float(rise[peak]), peak_step=int(steps[peak]),
                    first_below_3cm_after_peak=int(steps[down[0]]) if len(down) else None,
                    min_eef_target_m=float(gap.min()), closest_eef_step=int(steps[np.argmin(gap)]),
                    min_basket_xy_m=float(xy.min()), closest_basket_step=int(steps[np.argmin(xy)]),
                    final_basket_xy_m=float(xy[-1]), final_rise_m=float(rise[-1]),
                    min_eef_basket_xy_m=float(eef_basket_xy.min()),
                    first_eef_near_basket_object_far_step=int(steps[unsupported_near[0]]) if len(unsupported_near) else None,
                    eef_z_at_peak=float(eef[peak,2]), object_z_at_peak=float(obj[peak,2]))
                if len(down):
                    record.update(eef_z_at_height_return=float(eef[down[0],2]),
                                  object_z_at_height_return=float(obj[down[0],2]),
                                  eef_target_gap_at_height_return_m=float(gap[down[0]]))
                else:
                    record.update(eef_z_at_height_return=None, object_z_at_height_return=None,
                                  eef_target_gap_at_height_return_m=None)
                records.append(record)
                base_y = 35+arm_index*350
                draw.text((6,base_y), f'{arm} success={info["success"]} first_success={info["first_success_step"]}', fill='black', font=font)
                for frame_number, requested in enumerate(range(0,601,50)):
                    x, y = frame_number%7, frame_number//7
                    eligible = np.flatnonzero(trace['image_steps'] <= min(requested,end))
                    idx = int(eligible[-1])
                    frame = Image.fromarray(trace['agentview_images'][idx,lane]).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    canvas.paste(frame,(x*150+5,base_y+38+y*155))
                    draw.text((x*150+5,base_y+20+y*155),f'step {trace["image_steps"][idx]}',fill='black',font=font)
        output = RUN / f'inspect_{case}_trial{trial:02d}.png'
        canvas.save(output)
        manifest.append(dict(case=case,trial=trial,control=row['control'],preview=str(output.relative_to(ROOT))))
    with (OUT/'focused_events.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(records)
    (OUT/'FOCUSED_INSPECTION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__ == '__main__':
    main()
