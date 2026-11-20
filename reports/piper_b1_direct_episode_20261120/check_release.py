"""Check the block location and clearance from the gripper in saved B1 images."""
import json
import sys
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np


def files(episode):
    folder = episode / "episode0/camera/color/overhead"
    return sorted((p for p in folder.iterdir() if p.stem.replace(".", "").isdigit()), key=lambda p: Decimal(p.stem))


def region_mask(hsv, role):
    h, s, v = cv2.split(hsv)
    y, x = np.indices(h.shape)
    if role == "blue":
        return (h >= 88) & (h <= 135) & (s > 25) & (v > 52) & (x < 310) & (y > 100) & (y < 335)
    return ((h < 12) | (h > 165)) & (s > 70) & (v > 65) & (x > 135) & (x < 450) & (y > 255) & (y < 445)


def fixed_masks(reference_episodes):
    votes = {"blue": [], "red": []}
    for ep in reference_episodes:
        paths = files(ep)
        for i in [16, 30, -10]:
            hsv = cv2.cvtColor(cv2.imread(str(paths[i])), cv2.COLOR_BGR2HSV)
            for role in votes:
                votes[role].append(region_mask(hsv, role))
    masks = {}
    for role, regions in votes.items():
        common = (np.mean(regions, axis=0) > .40).astype("uint8")
        contours, _ = cv2.findContours(common, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(common)
        cv2.fillConvexPoly(mask, cv2.convexHull(max(contours,key=cv2.contourArea)), 1)
        masks[role] = cv2.erode(mask, np.ones((5,5),np.uint8)).astype(bool)
    return masks


def candidate(hsv, mask, dark_distance):
    h,s,v = cv2.split(hsv)
    wood = (h>=5) & (h<=38) & (s>=22) & (s<165) & (v>44) & (v<228) & mask
    wood = cv2.morphologyEx(wood.astype("uint8"), cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    contours, _ = cv2.findContours(wood,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        (x,y),(w,h),_ = cv2.minAreaRect(c)
        if 240 <= area <= 2300 and min(w,h) >= 14 and max(w,h) <= 66 and min(w,h) >= .47*max(w,h) and area >= .58*w*h:
            dist = float(dark_distance[int(round(y)),int(round(x))])
            if best is None or area > best[0]:
                best = (round(area), x,y,dist)
    return best


def stable_tail(candidates):
    last = candidates[-1]
    if not last:
        return None
    count = 0
    for c in reversed(candidates):
        if not c or np.hypot(c[1]-last[1],c[2]-last[2]) > 8:
            break
        count += 1
    tail = candidates[-count:]
    return dict(frames=count, min_clearance_px=round(min(c[3] for c in tail),1),
                end_clearance_px=round(last[3],1), end_block_area_px=last[0])


def release_windows(candidates, widths, seconds):
    # Keep stationary intervals with a visible gap to a large dark component.
    # Distances use the candidate centre, so subtract its equivalent radius.
    result = []
    start = None
    for i, c in enumerate([*candidates, None]):
        clear = c is not None and c[3] - np.sqrt(c[0]/np.pi) >= 6
        if start is not None:
            first = candidates[start]
            if not clear or np.hypot(c[1]-first[1],c[2]-first[2]) > 8:
                if i-start >= 3:
                    result.append(dict(first=start, last=i-1, frames=i-start,
                                       seconds=[round(float(seconds[j]),2) for j in (start,i-1)],
                                       width_mm=round(float(np.median(widths[start:i])),1)))
                start = None
        if start is None and clear:
            start = i
    return result


def inspect(episode, masks):
    rows = [json.loads(s) for s in (episode/"alignment.jsonl").read_text().splitlines()]
    paths = files(episode)
    assert len(paths) == len(rows)
    t = np.array([r["sample_monotonic_ns"] for r in rows],dtype="i8")
    seconds = (t-t[0])/1e9
    widths = np.array([r["feedback"]["gripper_mm"] for r in rows])
    minima = np.flatnonzero(widths <= widths.min()+.5)
    post = widths[minima[0]:]
    out = dict(open_after_min_mm=round(float(post.max()-widths.min()),1))
    detections = {role:[] for role in masks}
    for im in paths:
        hsv = cv2.cvtColor(cv2.imread(str(im)),cv2.COLOR_BGR2HSV)
        # Large dark regions include the black gripper, arm and their shadows.
        dark = (hsv[:,:,2] < 58).astype("uint8")
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
        selected = np.flatnonzero(stats[1:,cv2.CC_STAT_AREA] >= 90)+1
        distance = cv2.distanceTransform((~np.isin(labels,selected)).astype("uint8"),cv2.DIST_L2,5)
        for role, mask in masks.items():
            detections[role].append(candidate(hsv,mask,distance))
    for role, candidates in detections.items():
        present = np.array([bool(c) for c in candidates])
        indices = np.flatnonzero(present)
        runs = np.split(indices, np.flatnonzero(np.diff(indices)>1)+1)
        # Return counts and relative times, with no image or trajectory export.
        runs = [[int(g[0]),int(g[-1]),round(float(seconds[g[0]]),2),round(float(seconds[g[-1]]),2)] for g in runs if len(g)>=5]
        out[role] = dict(first30=int(present[:30].sum()), last15=int(present[-15:].sum()),
                         last30=int(present[-30:].sum()), total=int(present.sum()),runs=runs,
                         stable_tail=stable_tail(candidates))
        if role == "blue":
            out[role]["clear_intervals"] = release_windows(candidates, widths, seconds)
            for run in runs:
                group = candidates[run[0]:run[1]+1]
                # Range of measured gripper opening while the block is visible in the dish.
                w = widths[run[0]:run[1]+1]
                run.extend([round(float(w.min()),1), round(float(w.max()),1),
                            round(max(c[3]-np.sqrt(c[0]/np.pi) for c in group),1)])
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1])/"recordings"
    references = [next(root.glob("20261120_"+t+"_*")) for t in ("155508","155536","155612")]
    print(json.dumps(inspect(root/sys.argv[2],fixed_masks(references))))
