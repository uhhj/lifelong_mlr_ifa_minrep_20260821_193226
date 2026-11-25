from pathlib import Path
import sys, json
import numpy as np
ROOT = Path(sys.argv[1])
NAMES = ['20261125_132907_3026012_B1-ode50-aligned-20261125', '20261125_133047_3026715_B1-ode50-aligned-20261125']

def records(p):
    return [json.loads(s) for s in p.read_bytes().splitlines()]
cal = ROOT / 'offline_diagnostics/b1_spatial_ablation_20261121_2034'
sys.path.insert(0, str(cal))
from fit_gripper import flange
tip = np.array(json.loads((cal / 'gripper_calibration.json').read_bytes())['tcp_flange_mm'])

def tcp(q):
    f = flange(np.rad2deg(np.asarray(q)[..., :6]))
    return f[..., :3, 3] + f[..., :3, :3] @ tip

def describe(a):
    return dict(min=float(np.min(a)), median=float(np.median(a)), max=float(np.max(a)))
diagnostics = ROOT / 'offline_diagnostics'
selection = json.loads((diagnostics / 'b1_ode_lerobot_20261123/selection.json').read_bytes())
experts = []
for e in selection['ranking']:
    if e['id'] not in selection['selected_ids']:
        continue
    src = ROOT / 'batches/B1_20261123' / e['source']
    rows = records(src / 'alignment.jsonl')[e['first']:]
    if rows[0]['target']['gripper_mm'] <= 55:
        continue
    cm = np.array([r['target']['gripper_mm'] for r in rows])
    q = np.array([r['feedback']['joints_rad'] + [r['feedback']['gripper_mm']] for r in rows])
    xyz = tcp(q)
    close = np.flatnonzero(cm <= 55)
    if not len(close):
        continue
    k = close[0]
    crossed = np.flatnonzero((np.arange(len(q)) >= k) & (q[:, 6] <= 42))
    if not len(crossed):
        continue
    m = crossed[0]
    ts = np.array([r['sample_monotonic_ns'] for r in rows])
    experts.append(dict(e=e, src=src, k=k, m=m, q=q, xyz=xyz, ts=ts))
comparison = []
for name in NAMES:
    allrows = records(ROOT / 'policy_runs' / (name + '.jsonl'))
    rows = [r for r in allrows if r.get('mode') == 'execute']
    k = next((i for i, r in enumerate(rows) if r['sent'][6] <= 55))
    h = next((i for i, r in enumerate(rows) if r['gripper_control']['phase'] == 'waiting'))
    t = next((r['mono_ns'] for r in allrows if r['event'] == 'start'))
    xyz = tcp([r['measured_at_send'] for r in rows])
    inputs = records(ROOT / 'policy_runs' / name / 'inputs.jsonl')
    inp = {r['stamp']: r for r in inputs if r['mode_at_query'] == 'execute'}
    last_corrected = next((i for i in reversed(range(len(rows))) if inp[rows[i]['stamp']]['plan']['alignment']['applied']))
    dist = lambda e: float(np.mean(np.square(np.rad2deg(np.asarray(rows[k]['measured_at_send'][:6]) - e['q'][e['k'], :6]))) ** 0.5)
    examples = []
    for e in sorted(experts, key=dist)[:3]:
        frames = {}
        for key in ('k', 'm'):
            stamp = e['ts'][e[key]]
            im = e['src'] / 'episode0/camera/color/wrist' / f'{stamp / 1000000000.0:.6f}.jpg'
            assert im.is_file()
            frames[key] = str(im.relative_to(ROOT))
        examples.append(dict(episode=e['e']['id'], joint_rms_deg=dist(e), frames=frames))
    geometry = {}
    for label, i in [('start', 0), ('last_corrected', last_corrected), ('first_sent_55', k), ('waiting', h), ('end', len(rows) - 1)]:
        r = rows[i]
        geometry[label] = dict(offset_s=(r['mono_ns'] - t) / 1000000000.0, xyz_mm=xyz[i].tolist(), measured_mm=r['measured_at_send'][6])
    matched = []
    for label, i in [('start', 0), ('first_sent_55', k), ('wait', h)]:
        candidates = []
        field_q = np.array(rows[i]['measured_at_send'][:6])
        for e in experts:
            dif = np.rad2deg(e['q'][:, :6] - field_q)
            rms = (dif ** 2).mean(axis=1) ** 0.5
            j = int(rms.argmin())
            candidates.append((rms[j], e['e']['id'], j, e, dif[j]))
        rms, episode, j, e, diff = min(candidates, key=lambda item: item[0])
        rel = e['src'] / 'episode0/camera/color/overhead' / f"{e['ts'][j] / 1000000000.0:.6f}.jpg"
        matched.append(dict(phase=label, episode=episode, joint_rms=float(rms), max_joint=float(np.abs(diff).max()), xyz_separation_mm=float(np.linalg.norm(e['xyz'][j] - xyz[i])), expert_width=e['q'][j, 6], frame=str(rel.relative_to(ROOT)), field_frame=inp[rows[i]['stamp']]['images']['scene']))
    comparison.append(dict(name=name, geometry=geometry, examples=examples, nearest_posture=matched))
print(json.dumps(dict(count=len(experts), expert_z_first_cmd55=describe([e['xyz'][e['k'], 2] for e in experts]), expert_z_first_feedback42=describe([e['xyz'][e['m'], 2] for e in experts]), trials=comparison)))
