"""Compare lateral motion with complete, recorded B1/B2 demonstration trajectories."""
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from math import degrees
from pathlib import Path
from statistics import median
import hashlib
import json
import sys


def stats(values):
    return [round(f(values),3) for f in (min,median,max)]


work=Path(sys.argv[1])
path=work/"policy_runs"/sys.argv[2]
session=[json.loads(s) for s in path.read_bytes().splitlines()]
start=next(r for r in session if r["event"]=="start")
run=[r for r in session if r.get("mode")=="execute" and r["event"]=="prediction"]
field=[degrees(r["measured_at_send"][0]) for r in run]
field_start=[degrees(v) for v in start["initial"][:6]]+[start["initial"][6]]
summary={}
sets=(("B1","B1_top50_20261120_163349","10fc15aff94bc6deb746c4f838ad04fef8156cddaea153b49301376aec3c38cb"),
      ("B2","B2_top50_20261120_173919","481469382a4725d659874b33575bf9a254deff91b7292c8509431a06271d7dc7"))
for task,name,manifest_sha in sets:
    base=work/"expert_sets"/name
    manifest=(base/"manifest.json").read_bytes()
    assert hashlib.sha256(manifest).hexdigest()==manifest_sha
    entries=json.loads(manifest)["episodes"]
    assert len(entries)==50
    ref=[]
    for e in entries:
        raw=(base/e["episode_dir"]/"alignment.jsonl").read_bytes()
        assert hashlib.sha256(raw).hexdigest()==e["alignment_sha256"]
        rows=[json.loads(s) for s in raw.splitlines()][e["first_action_row"]:e["last_action_row"]+1]
        ts=[(r["sample_monotonic_ns"]-rows[0]["sample_monotonic_ns"])/1e9 for r in rows]
        q=[r["feedback"] for r in rows]
        j=[degrees(f["joints_rad"][0]) for f in q]
        grip=[f["gripper_mm"] for f in q]
        ref.append(dict(start=[degrees(v) for v in q[0]["joints_rad"]]+[grip[0]],min_j1=min(j),
                        max_j1=max(j),min_target_j1=min(degrees(r["target"]["joints_rad"][0]) for r in rows),
                        delta_j1=[j[bisect_left(ts,s)]-j[0] if ts[-1]>=s else None for s in (1,2,3,5)],
                        grip_min=min(grip),grip_max=max(grip),grip_drop=grip[0]-min(grip),
                        final_j1=j[-1],first_above_2=next((t for t,v in zip(ts,j) if v-j[0]>=2),None)))
    min_all=min(ep["min_j1"] for ep in ref)
    min_target=min(ep["min_target_j1"] for ep in ref)
    bounds=[max(x)-min(x) for x in zip(*(e["start"] for e in ref))]
    near=sorted(ref,key=lambda ep:sum(((v-a)/b)**2 for v,a,b in zip(field_start,ep["start"],bounds)))[:5]
    def phase(episodes):
        return {str(sec):dict(delta_j1=stats(vals), negative=sum(v<0 for v in vals),n=len(vals))
                for k,sec in enumerate((1,2,3,5)) if (vals:=[e["delta_j1"][k] for e in episodes
                                                              if e["delta_j1"][k] is not None])}
    def crossing(series,lo):
        idx=next((i for i,v in enumerate(series) if v<lo),None)
        if idx is None:return None
        return round((run[idx]["mono_ns"]-start["mono_ns"])/1e9,3)
    summary[task]=dict(n=len(ref),per_episode_min_j1=stats([e["min_j1"] for e in ref]),
                       per_episode_max_j1=stats([e["max_j1"] for e in ref]),per_episode_min_grip=stats([e["grip_min"] for e in ref]),
                       final_j1=stats([e["final_j1"] for e in ref]), grip_drop_from_start=stats([e["grip_drop"] for e in ref]),
                       field_first_below_all_j1_s=crossing(field,min_all), min_expert_target_j1=round(min_target,3),
                       field_target_first_below_all_target_j1_s=crossing([degrees(r["sent"][0]) for r in run],min_target),
                       phase=phase(ref), five_nearest_start_phase=phase(near))
files=[p.suffix for p in path.parent.iterdir() if p.stem==path.stem]
print(json.dumps(dict(reference=summary,execution_file_types=sorted(files),
                      logged_prediction_fields=sorted(run[0]),task=session[0].get("task")),ensure_ascii=False))
