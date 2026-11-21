"""Compare one ended B1 run with verified archived expert positions and commands.

Arguments are the field run directory (with main.jsonl) and the B1 expert-set
directory or a preverified private gzip export. Add --plan to emit replay pairs.
"""
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


# Initial overhead images were visually inspected, before examining these trajectories.
# The block touches or straddles the right edge of the red paper in these examples.
SCENE_IDS = (
    "20261120_153607_4138658", "20261120_155638_4146536",
    "20261120_155959_4149320", "20261120_160932_4158084",
)


def degrees_mm(values):
    values = np.asarray(values, dtype=float).copy()
    values[:, :6] = np.degrees(values[:, :6])
    return values


def span(values):
    values = np.asarray(values)
    return dict(n=len(values), min_med_max=np.round(np.percentile(values, [0, 50, 100]), 3).tolist())


def crossing(values, level):
    idx = np.flatnonzero(values <= level)
    return int(idx[0]) if len(idx) else None


def expert_episodes(source):
    if source.is_file():
        return json.loads(gzip.decompress(source.read_bytes()))["episodes"]
    manifest=(source/"manifest.json").read_bytes()
    assert hashlib.sha256(manifest).hexdigest()=="10fc15aff94bc6deb746c4f838ad04fef8156cddaea153b49301376aec3c38cb"
    episodes=[]
    for ep in sorted(json.loads(manifest)["episodes"],key=lambda e:e["episode_id"]):
        raw=(source/ep["episode_dir"]/"alignment.jsonl").read_bytes()
        assert hashlib.sha256(raw).hexdigest()==ep["alignment_sha256"]
        rows=[json.loads(s) for s in raw.splitlines()][ep["first_action_row"]:ep["last_action_row"]+1]
        def vectors(key):
            return [r[key]["joints_rad"]+[r[key]["gripper_mm"]] for r in rows]
        episodes.append(dict(id=ep["episode_id"],t=[r["sample_monotonic_ns"] for r in rows],
                             q=vectors("feedback"),a=vectors("target")))
    return episodes


def analyze(field, expert_export):
    field_rows = [json.loads(line) for line in (field/"main.jsonl").read_bytes().splitlines()]
    assert field_rows[-1]["event"] == "end"
    start = next(r for r in field_rows if r["event"] == "start")
    run = [r for r in field_rows if r["event"] == "prediction" and r.get("mode") == "execute"]
    t = (np.array([r["mono_ns"] for r in run])-start["mono_ns"])*1e-9
    pred, cmd, obs, q_image = [degrees_mm([r[key] for r in run]) for key in ("p","sent","measured_at_send","q")]
    episodes = expert_episodes(expert_export)
    assert len(episodes) == len({e["id"] for e in episodes}) == 50
    q0 = degrees_mm([start["initial"]])[0]
    initial = degrees_mm([e["q"][0] for e in episodes])
    ranges = np.ptp(initial, axis=0)
    nearest = np.argsort(np.linalg.norm((initial-q0)/np.where(ranges>0,ranges,1),axis=1))[:5]
    nearest_ids = {episodes[i]["id"] for i in nearest}
    p = np.flatnonzero(t>4.5)
    trough = p[np.argmin(cmd[p,4])]
    feedback_trough = p[np.argmin(obs[p,4])]
    last = len(t)-1
    begin_ramp = np.argmin(abs(t-(t[-1]-.337)))
    ramp_offsets = t[[begin_ramp,last]]-t[trough]
    residual = pred-q_image
    bracket = (t>=2)&(t<4)
    j1min = int(np.argmin(cmd[:,0]))
    field_summary = dict(
        frames=len(run), first_j1_below_minus1_s={
            k:round(float(t[crossing(v[:,0]-q0[0],-1)]),3)
            for k,v in (("model",pred),("sent",cmd),("feedback",obs))},
        j1_minimum_sent_deg=round(cmd[j1min,0],3), j1_minimum_time_s=round(t[j1min],3),
        negative_j1_corrections_2_to_4_s=int(np.sum(residual[bracket,0]<0)),frames_2_to_4_s=int(bracket.sum()),
        sent_j1_final_deg=round(cmd[-1,0],3), j5_minimum_command_s=round(t[trough],3),
        j5_minimum_feedback_s=round(t[feedback_trough],3),
        j5_ramp=dict(interval_s=[round(t[i],4) for i in (begin_ramp,last)],
                     target_deg=[round(cmd[i,4],3) for i in (begin_ramp,last)],
                     feedback_deg=[round(obs[i,4],3) for i in (begin_ramp,last)]),
        j1_delta_timed={str(s): round(np.interp(s,t,obs[:,0])-q0[0],3) for s in (2,3,4,5)},
        grip_events={}, j5_raw_minus_q_peak_after_trough_deg=round(max(residual[trough:,4]),3))
    metrics = []
    plan = {}
    for ep in episodes:
        ts = (np.array(ep["t"])-ep["t"][0])*1e-9
        assert np.all(np.diff(ts)>0)
        q,a = degrees_mm(ep["q"]),degrees_mm(ep["a"])
        item = dict(id=ep["id"], j1_delta_timed={str(s):float(np.interp(s,ts,q[:,0])-q[0,0]) for s in (2,3,4,5)},
                    grip_events={},grip_min=float(min(q[:,6])),j5_reversal=None)
        for level in (55,50,43):
            j = crossing(q[:,6],level)
            if j is not None and j>0:
                item["grip_events"][str(level)] = [ts[j], q[j,0]-q[0,0],q[j,4],a[j,4]-q[j,4],q[j,1]]
        # Measure a down-to-up reversal near closure, independent of absolute trial duration.
        # Same rule on real and expert targets; both sides must exceed 5 deg over 0.8 s.
        closure = crossing(q[:,6],50)
        if closure is not None:
            window = np.flatnonzero((abs(ts-ts[closure])<=1.5)&(ts>=.8)&(ts+max(.8,ramp_offsets[-1])<=ts[-1]))
            candidates=[]
            for j in window:
                if a[j,4] > min(a[abs(ts-ts[j])<=.18,4]) + 1e-8:
                    continue
                before,after = np.interp(ts[j]+np.array([-.8,.8]),ts,a[:,4])-a[j,4]
                if min(before,after)>=5:
                    candidates.append(j)
            if candidates:
                j = min(candidates,key=lambda i:abs(ts[i]-ts[closure]))
                sample = ramp_offsets+ts[j]
                def rate(series):
                    start_value,end_value=np.interp(sample,ts,series)
                    return float((end_value-start_value)/(sample[1]-sample[0]))
                item["j5_reversal"] = dict(target_rate=rate(a[:,4]),feedback_rate=rate(q[:,4]),
                    lag=float(np.interp(sample[-1],ts,a[:,4]-q[:,4])))
        metrics.append(item)
        if ep["id"] in SCENE_IDS:
            # Match the two first descending gripper crossings; interpolation sets the reference phase.
            anchors = [crossing(q[:,6],w) for w in (55,50)]
            assert None not in anchors and anchors[0]<=anchors[1]
            field_anchors = [t[crossing(obs[:,6],w)] for w in (55,50)]
            picks = []
            for sec in (2,2.5,3,3.5,4,4.5,5,5.25,5.5,5.8):
                fidx = int(np.argmin(abs(t-sec)))
                et = np.interp(t[fidx],[0,*field_anchors],[0,ts[anchors[0]],ts[anchors[1]]])
                if t[fidx]>field_anchors[-1]:
                    et = ts[anchors[1]]+(t[fidx]-field_anchors[-1])
                ej = int(np.argmin(abs(ts-et)))
                picks.append([int(run[fidx]["stamp"]),ej])
            plan[ep["id"]]=picks
    groups = {}
    for name,subset in (("all",metrics),("similar_block",[m for m in metrics if m["id"] in SCENE_IDS]),
                        ("similar_start",[m for m in metrics if m["id"] in nearest_ids])):
        group = {}
        for sec in (2,3,4,5):
            data=[m["j1_delta_timed"][str(sec)] for m in subset]
            group[str(sec)]=dict(j1=span(data),negative=int(sum(v<0 for v in data)))
        events = {}
        for w in ("55","50","43"):
            event=[m["grip_events"][w] for m in subset if w in m["grip_events"]]
            if event:
                events[w] = {k:span(col) for k,col in zip(("time","j1_delta","j5_angle","j5_target_minus_feedback","j2_angle"),zip(*event))}
        reversals=[m["j5_reversal"] for m in subset if m["j5_reversal"]]
        groups[name]=dict(n=len(subset),timed=group,grip=events,
                           reversal={k:span([r[k] for r in reversals]) for k in ("target_rate","feedback_rate","lag") if reversals},
                           reversal_count=len(reversals),min_grip_mm=span([m["grip_min"] for m in subset]))
    for width in (55,50,43):
        idx = crossing(obs[:,6],width)
        if idx is not None:
            field_summary["grip_events"][str(width)]=dict(t_s=round(t[idx],3),j1_delta=round(obs[idx,0]-q0[0],3),
                j5_angle=round(obs[idx,4],3),j2_angle=round(obs[idx,1],3),j2_target=round(cmd[idx,1],3),
                j2_max_prediction=round(max(pred[:idx+1,1]),3))
    return dict(field=field_summary,experts=groups),plan


if __name__ == "__main__":
    result, plan = analyze(Path(sys.argv[1]),Path(sys.argv[2]))
    print(json.dumps(plan if "--plan" in sys.argv[3:] else result,indent=2,allow_nan=False))
