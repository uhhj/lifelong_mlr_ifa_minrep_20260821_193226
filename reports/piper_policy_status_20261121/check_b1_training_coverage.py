"""Compare the field start and closing event to training examples, using held-out episodes as a reference."""
import json
from pathlib import Path
import random
import sys

import numpy as np

from compare_b1_pickup_phase import SCENE_IDS, degrees_mm, expert_episodes


def compare_to_train(field, train, held_out, scale):
    def distance(reference, query):
        return np.sqrt(np.mean(((reference-query)/scale)**2,axis=1))

    field_distances=distance(train,field)
    nearest=train[np.argmin(field_distances)]
    validation_distances=[min(distance(train,v)) for v in held_out]
    return dict(train=len(train),held_out=len(held_out),
                field_min_RMS=round(float(min(field_distances)),4),
                held_out_min_RMS_range_median=np.round(np.percentile(validation_distances,[0,50,100]),4).tolist(),
                held_out_at_least_as_far=int(np.sum(np.array(validation_distances)>=min(field_distances))),
                nearest_absolute_axis_differences_deg_mm=np.round(abs(nearest-field),3).tolist())


def audit(field_dir, expert_source):
    experts=expert_episodes(expert_source)
    assert len(experts)==50 and len({e["id"] for e in experts})==50
    validation=set(random.Random(42).sample(sorted(e["id"] for e in experts),10))
    train,held_out=([degrees_mm(e["q"]) for e in experts if (e["id"] in validation)==flag] for flag in (False,True))
    assert (len(train),len(held_out))==(40,10)
    # Standardize each joint and the gripper using only B1 training feedback.
    scale=np.concatenate(train).std(axis=0)
    assert np.all(scale>0)
    log=[json.loads(line) for line in (field_dir/"main.jsonl").read_bytes().splitlines()]
    assert log[-1]["event"]=="end"
    start=next(r for r in log if r["event"]=="start")
    run=[r for r in log if r["event"]=="prediction" and r.get("mode")=="execute"]
    times=(np.array([r["mono_ns"] for r in run])-start["mono_ns"])/1e9
    actual=degrees_mm([r["measured_at_send"] for r in run])
    predicted=degrees_mm([r["p"] for r in run])
    sent=degrees_mm([r["sent"] for r in run])
    field0=degrees_mm([start["initial"]])[0]
    def closing_indices(sequence):
        return np.flatnonzero((sequence[:-1,6]>50)&(sequence[1:,6]<=50))+1

    def first_closures(episodes):
        # Exclude the episode already below threshold at the first action row.
        return np.stack([q[closing_indices(q)[0]] for q in episodes if q[0,6]>50 and len(closing_indices(q))])

    closure=closing_indices(actual)[0]
    train_event=first_closures(train)
    val_event=first_closures(held_out)
    peak=np.argmax(predicted[:closure+1,1])
    def point(i):
        return dict(t_s=round(times[i],3),j2_raw_deg=round(predicted[i,1],3),
                    j2_sent_deg=round(sent[i,1],3),j2_actual_deg=round(actual[i,1],3),
                    gripper_actual_mm=round(actual[i,6],2))

    scene_train=sorted(set(SCENE_IDS)-validation)
    paired=[]
    for ep in experts:
        if ep["id"] in scene_train:
            q=degrees_mm(ep["q"])
            at=closing_indices(q)[0]
            paired.append(dict(start_j2_deg=round(q[0,1],3),close_j2_deg=round(q[at,1],3)))
    return dict(
        similar_block_train=len(scene_train),similar_block_validation=len(set(SCENE_IDS)&validation),
        similar_block_train_j2=paired,field_initial=field0.round(3).tolist(),
        start_comparison=compare_to_train(field0,np.stack([q[0] for q in train]),[q[0] for q in held_out],scale),
        close_comparison=compare_to_train(actual[closure],train_event,val_event,scale),
        training_close_j2_deg_range=np.round([min(train_event[:,1]),max(train_event[:,1])],3).tolist(),
        peak_j2_before_close=point(peak),first_closure=point(closure))


if __name__=="__main__":
    print(json.dumps(audit(Path(sys.argv[1]),Path(sys.argv[2])),indent=2,allow_nan=False))
