"""Render sparse recorded frames for the first discordant trial per fixed comparison."""
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
    examples = []
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    for case in ('seed43_stage7_task7', 'seed42_stage9_task7', 'seed43_stage9_task9'):
        for control in ('mlr', 'release_ifa'):
            candidates = sorted([r for r in rows if r['case'] == case and r['control'] == control
                                 and r['corrected_success'] != r['control_success']], key=lambda r: int(r['trial']))
            if not candidates:
                continue
            row = candidates[0]
            trial = int(row['trial'])
            round_id, lane = divmod(trial, 10)
            canvas = Image.new('RGB', (6*150, 2*170+42), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 5), f'{case} / trial {trial} / first discordant vs {control}', fill='black', font=font)
            for y, arm in enumerate((control, 'corrected_ifa')):
                path = RUN / case / arm
                trials = json.loads((path / f'round{round_id}.json').read_text())
                info = trials[lane]
                end = info['first_success_step'] or info['policy_steps']
                with np.load(path / f'round{round_id}.npz') as trace:
                    available = trace['image_steps']
                    for x, requested in enumerate((0, 50, 100, 200, 400, 600)):
                        indices = np.flatnonzero(available <= min(requested, end))
                        idx = int(indices[-1])
                        # Display correction only; raw recorded policy input remains unchanged.
                        frame = Image.fromarray(trace['agentview_images'][idx, lane]).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                        canvas.paste(frame, (x*150+6, y*170+70))
                        draw.text((x*150+6, y*170+47), f'step {available[idx]}', fill='black', font=font)
                draw.text((6, y*170+201), f'{arm}: success={info["success"]}, first success={info["first_success_step"]}', fill='black', font=font)
            image = RUN / f'{case}_vs_{control}_trial{trial}.png'
            canvas.save(image)
            examples.append(dict(**row, preview=str(image.relative_to(ROOT)),
                                 selection='first increasing-index discordant trial; frames at/before fixed step grid and before first success'))
    (OUT / 'EXAMPLE_MANIFEST.json').write_text(json.dumps(examples, indent=2)+'\n')
    print(json.dumps(examples, indent=2))


if __name__ == '__main__':
    main()
