from pathlib import Path
import sys, json, struct, collections
import numpy as np
w = Path(sys.argv[1])
old = w.parent / 'diagnostics/piper_passive_20261119_195013_th1n18bz'
dtype = np.dtype([('wall', '<u8'), ('mono', '<u8'), ('cid', '<u4'), ('port', 'u1'), ('dlc', 'u1'), ('flags', '<u2'), ('data', 'V8')])
chunks = []
for f in sorted(old.glob('frames-*.bin')):
    n = f.stat().st_size // dtype.itemsize
    if n:
        a = np.memmap(f, dtype=dtype, mode='r', shape=(n,))
        chunks.append((int(a['mono'][0]), int(a['mono'][-1]), a))
ep = list(next((w / 'expert_sets').glob('B1_top50*')).joinpath('episodes').iterdir())
groups = collections.defaultdict(list)
eps = []
faults = set()
matched_before_lift = 0
matched_any = 0
wide_matches = 0
cal = w / 'offline_diagnostics/b1_spatial_ablation_20261121_2034'
sys.path.insert(0, str(cal))
from fit_gripper import flange
tip = np.array(json.loads((cal / 'gripper_calibration.json').read_bytes())['tcp_flange_mm'])
for d in sorted(ep):
    f = d / 'alignment.jsonl'
    if not f.is_file():
        continue
    rows = [json.loads(x) for x in f.read_bytes().splitlines()]
    lo = rows[0]['sample_monotonic_ns'] - 200000000
    hi = rows[-1]['sample_monotonic_ns'] + 200000000
    samples = []
    anchors = collections.defaultdict(list)
    for first, last, a in chunks:
        if first > hi or last < lo:
            continue
        l, r = a['mono'].searchsorted([lo, hi])
        v = a[l:r]
        for z in v[(v['port'] == 0) & (v['dlc'] == 8) & np.isin(v['cid'], [341, 680])]:
            b = z['data'].tobytes()
            wall = int(z['wall'])
            if z['cid'] == 341:
                anchors[b].append(wall)
            else:
                samples.append((wall, b))
    offsets = collections.Counter()
    for r in rows:
        t = r['target']
        if t and len(anchors[bytes.fromhex(t['frame_hex'][0])]) == 1:
            offsets[anchors[bytes.fromhex(t['frame_hex'][0])][0] - t['joint_bus_monotonic_ns'][0]] += 1
    if not offsets:
        continue
    offset, count = offsets.most_common(1)[0]
    m = dict(samples)
    vals = []
    for r in rows:
        q = r['feedback']
        t = r['target']
        if not q or not t:
            continue
        b = m.get(q['component_monotonic_ns'][-1] + offset)
        if not b:
            continue
        width, tau, status = struct.unpack_from('>ihB', b)
        if abs(q['gripper_mm'] - width / 1000) > 1e-05:
            continue
        vals.append((q['joints_rad'], width / 1000, tau / 1000, t['gripper_mm'], q['component_monotonic_ns'][-1], status))
        faults.add(status)
    if len(vals) < 40:
        continue
    pose = flange(np.rad2deg([v[0] for v in vals]))
    z = (pose[:, :3, 3] + pose[:, :3, :3] @ tip)[:, 2]
    widths = np.array([v[1] for v in vals])
    tau = np.array([v[2] for v in vals])
    cmd = np.array([v[3] for v in vals])
    below = np.where(widths <= 48)[0]
    if not len(below):
        continue
    k = below[0]
    raised = (z > z[k] + 25) & (np.arange(len(z)) > k) & (cmd + 2 < widths) & (widths >= 25) & (widths < 52)
    out = np.flatnonzero((np.arange(len(z)) > k) & (widths >= 54))
    if len(out):
        raised[out[0]:] = False
    free = (widths >= 56) & (np.arange(len(z)) < k)
    if raised.sum() < 3 or free.sum() < 3:
        continue
    matches = []
    for i, v in enumerate(vals):
        if not (35 <= v[1] <= 44 and -1.2 <= v[2] <= -0.8 and (v[1] - v[3] >= 1)):
            continue
        j = i
        while j > 0 and v[4] - vals[j][4] < 150000000:
            j -= 1
        win = vals[j:i + 1]
        if len(win) >= 5 and v[4] - win[0][4] >= 150000000 and all((0 < b[4] - a[4] <= 50000000 for a, b in zip(win, win[1:]))) and (max((x[1] for x in win)) - min((x[1] for x in win)) <= 0.6) and all((x[5] == 64 and 35 <= x[1] <= 44 and (-1.2 <= x[2] <= -0.8) for x in win)):
            matches.append(i)
    matched_any += int(bool(matches))
    matched_before_lift += int(any((k <= i < np.flatnonzero(raised)[0] for i in matches)))
    wide_matches += sum((bool(free[i]) for i in matches))
    for label, ar in [('grasp_tau', tau[raised]), ('grasp_width', widths[raised]), ('free_tau', tau[free]), ('pre_motion_tau', tau[free & (cmd + 2 < widths)])]:
        groups[label].extend(ar)
    eps.append([float(np.median(tau[raised])), float(np.median(widths[raised])), float(np.max(tau[free]))])

def s(v):
    return dict(n=len(v), quantiles=np.round(np.percentile(v, [0, 5, 25, 50, 75, 95, 100]), 3).tolist()) if len(v) else None
new_width = []
sel = json.loads((w / 'offline_diagnostics/b1_ode_lerobot_20261123/selection.json').read_bytes())
for e in sel['ranking']:
    if e['id'] not in sel['selected_ids']:
        continue
    rs = [json.loads(x) for x in (w / 'batches/B1_20261123' / e['source'] / 'alignment.jsonl').read_bytes().splitlines()][e['first']:]
    widths = np.array([r['feedback']['gripper_mm'] for r in rs])
    cmd = np.array([r['target']['gripper_mm'] for r in rs])
    pose = flange(np.rad2deg([r['feedback']['joints_rad'] for r in rs]))
    z = (pose[:, :3, 3] + pose[:, :3, :3] @ tip)[:, 2]
    below = np.flatnonzero(widths <= 48)
    if not len(below):
        continue
    k = below[0]
    raised = (z > z[k] + 25) & (np.arange(len(z)) > k) & (cmd + 2 < widths) & (widths >= 25) & (widths < 52)
    out = np.flatnonzero((np.arange(len(z)) > k) & (widths >= 54))
    if len(out):
        raised[out[0]:] = False
    if raised.sum() >= 3:
        new_width.append(float(np.median(widths[raised])))
print(json.dumps(dict(episodes=len(eps), candidates=len(ep), field_status=sorted(faults), quantile_percentages=[0, 5, 25, 50, 75, 95, 100], summary={k: s(v) for k, v in groups.items()}, per_episode_quantiles=[s(x) for x in zip(*eps)], new_grasp_width=s(new_width), contact_before_lift=matched_before_lift, contact_any=matched_any, wide_matches=wide_matches)))
