"""Define the table-coordinate target and evaluation pairs without reading any policy outputs."""
from pathlib import Path
import hashlib
import json
import sys
import networkx as nx
import numpy as np
from scipy.optimize import least_squares

from annotate import vectors


def reviewed(work,out):
    labels=json.loads((out/"points.json").read_bytes())
    manual=json.loads((out/"annotations.json").read_bytes())
    for k,v in manual.items():
        assert labels[int(k)]["index"]==int(k)
        labels[int(k)].update(v)
    base=work/"offline_diagnostics/b1_ode_lerobot_20261123"
    manifest=json.loads((base/"selection.json").read_bytes())
    ranked={r["id"]:r for r in manifest["ranking"]}
    records=[]
    for label in labels:
        r=ranked[label["episode"]]
        folder=work/"batches/B1_20261123"/r["source"]
        raw=(folder/"alignment.jsonl").read_bytes()
        assert hashlib.sha256(raw).hexdigest()==r["alignment_sha256"]
        rows=[json.loads(line) for line in raw.splitlines()][r["first"]:]
        records.append(dict(**label,folder=folder,first=r["first"],q=vectors(rows,"feedback"),cmd=vectors(rows,"target")))
    return records


def geometry(work,out):
    spatial=work/"offline_diagnostics/b1_spatial_ablation_20261121_2034"
    sys.path.insert(0,str(spatial))
    from fit_gripper import flange
    tip=np.asarray(json.loads((spatial/"gripper_calibration.json").read_bytes())["tcp_flange_mm"])
    def xy(q):
        transform=flange(np.rad2deg(np.asarray(q,dtype=np.float64)[...,:6]))
        return (transform[...,:3,3]+transform[...,:3,:3]@tip)[...,:2]
    rs=reviewed(work,out)
    train=[r for r in rs if r["train"] and r["contact"] is not None and r["uv"] is not None]
    def fit(items):
        uv=np.array([(*r["uv"],1.) for r in items])
        target=np.stack([xy(r["q"][r["contact"]]) for r in items])
        initial=np.linalg.lstsq(uv,target,rcond=None)[0]
        return least_squares(lambda v:(uv@v.reshape(3,2)-target).ravel(),initial.ravel(),loss="huber",f_scale=5).x.reshape(3,2)
    coef=fit(train)
    def error(r,matrix):
        return float(np.linalg.norm(np.r_[r["uv"],1.]@matrix-xy(r["q"][r["contact"]])))
    loo=[error(r,fit(train[:i]+train[i+1:])) for i,r in enumerate(train)]
    test=[r for r in rs if not r["train"] and r["close"] is not None and r["uv"] is not None
          and 7<=r["close"]-18 and r["close"]+5<=len(r["q"]) and r["contact"] is not None]
    goals={r["episode"]:np.r_[r["uv"],1.]@coef for r in test}
    graph=nx.Graph()
    for i,a in enumerate(test):
        ka=a["close"]-18
        for j,b in enumerate(test[:i]):
            kb=b["close"]-18
            q1,q2=a["q"][ka],b["q"][kb]
            separation=np.linalg.norm(goals[a["episode"]]-goals[b["episode"]])
            dist=np.linalg.norm(xy(q1)-xy(q2))
            angle=np.rad2deg(np.abs(q1[:6]-q2[:6]))
            if separation>=20 and dist<=55 and angle.max()<=25 and abs(q1[6]-q2[6])<=20:
                cost=dist/55+np.linalg.norm(angle)/25+abs(q1[6]-q2[6])/20
                graph.add_edge(j,i,weight=-float(cost)-1e-10*(i*i+j))
    pairs=sorted([sorted((test[a]["episode"],test[b]["episode"])) for a,b in nx.max_weight_matching(graph,maxcardinality=True)])
    contact_xy={r["episode"]:xy(r["q"][r["contact"]]) for r in test}
    data=dict(training_points=len(train),test_episodes=[r["episode"] for r in test],pairs=pairs,
              table_map=coef.tolist(),tip_mm=tip.tolist(),train_loo_error_mm_p50_p90=np.percentile(loo,[50,90]).tolist(),
              test_contact_error_mm_p50_p90=np.percentile([error(r,coef) for r in test],[50,90]).tolist(),
              test_uv_range=np.ptp(np.stack([r["uv"] for r in test]),axis=0).tolist(),
              test_base_xy_range_mm=np.ptp(np.stack(list(goals.values())),axis=0).tolist(),
              pair_separations_mm=[float(np.linalg.norm(goals[a]-goals[b])) for a,b in pairs],
              pair_expert_contact_direction_agreement=[bool((goals[a]-goals[b])@(contact_xy[a]-contact_xy[b])>0) for a,b in pairs])
    return data, test, goals, xy


if __name__=="__main__":
    work,out=map(Path,sys.argv[1:3])
    data,*_=geometry(work,out)
    (out/"geometry.json").write_text(json.dumps(data,indent=2))
    print(json.dumps({k:v for k,v in data.items() if k not in ("test_episodes","pairs","table_map")}))
