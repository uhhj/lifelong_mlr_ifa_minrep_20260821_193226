"""Compare two fixed table landmarks in six pairs of overhead frames."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def gray(path):
    with Image.open(path) as image:
        return np.asarray(image.crop((0, 0, 640, 480)).convert("L").resize((320, 240)), dtype=float)


def match(source, target, center, side, radius):
    x, y = (round(v/2) for v in center)
    patch = source[y-side:y+side+1, x-side:x+side+1].copy()
    patch -= patch.mean()
    a = max(0, y-radius-side)
    b = max(0, x-radius-side)
    region = target[a:y+radius+side+1, b:x+radius+side+1]
    tiles = np.lib.stride_tricks.sliding_window_view(region, patch.shape)
    shape = tiles.shape[:2]
    tiles = tiles.reshape(-1, patch.size)
    tiles = tiles - tiles.mean(axis=1, keepdims=True)
    scores = (tiles @ patch.ravel()) / np.maximum(np.linalg.norm(tiles, axis=1)*np.linalg.norm(patch), 1e-8)
    i, j = np.unravel_index(scores.argmax(), shape)
    return dict(expert_minus_field_px=[int(2*(j+b+side-x)), int(2*(i+a+side-y))],
                ncc=round(float(scores.max()),3))


def measure(root):
    results = []
    for trial_i, trial in enumerate(json.loads((root/"geometry.json").read_bytes())["trials"]):
        for i, ex in enumerate(trial["nearest_posture"]):
            actual = gray(root/trial["name"]/ex["field_frame"])
            reference = gray(root/("pose_"+trial["name"][9:15]+"_"+ex["phase"]+".jpg"))
            results.append(dict(trial=trial_i+1, phase=ex["phase"], joint_rms_deg=ex["joint_rms"],
                                estimated_tcp_difference_mm=ex["xyz_separation_mm"],
                                cable_hole=match(actual,reference,(505,175),12,12),
                                paper_corner=match(actual,reference,(485,442),14,12)))
    return results


if __name__ == "__main__":
    print(json.dumps(measure(Path(sys.argv[1])), indent=2))
