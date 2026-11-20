"""Pick review frames for the fixed B3 tabletop layout from saved synchronized images."""
from pathlib import Path
import sys
import cv2
import numpy as np


def files(episode, role):
    folder = episode / "episode0/camera/color" / role
    return sorted(p for p in folder.iterdir() if p.stem.replace(".", "").isdigit())


def review(episode):
    overhead, wrist = (files(episode, role) for role in ("overhead", "wrist"))
    assert overhead and [p.stem for p in overhead] == [p.stem for p in wrist]
    n = len(overhead)
    # Look for a clear view of the paper after the wrist has reached it.
    # These scores only select images. The decision is made by viewing them.
    candidates = sorted(set([*range(n // 3, n, 4), n - 1]))
    yellow = []
    for i in candidates:
        image = cv2.resize(cv2.imread(str(wrist[i])), (160, 120))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv[25:112])
        yellow.append(int(np.sum((h >= 17) & (h <= 40) & (s >= 49) & (v >= 60))))
    peak = candidates[int(np.argmax(yellow))]
    late = [i for i in candidates if i >= peak]
    def obstruction(i):
        img = cv2.imread(str(overhead[i]))[278:374, 120:225]
        return int(np.sum(np.max(img, axis=2) < 83))
    best = min(late, key=lambda i: (obstruction(i), -i))
    later = [i for i in late if i >= best + 5]
    second = min(later, key=lambda i: (obstruction(i), -i)) if later else best
    chosen = [0, int(.42*(n-1)), int(.62*(n-1)), peak, best, second, n-1]
    tiles = []
    for k,i in enumerate(chosen):
        if k == 0:
            im = cv2.imread(str(overhead[i]))[260:460, 80:560]
        elif k in (1, 2, 3):
            im = cv2.imread(str(wrist[i]))
        else:
            im = cv2.imread(str(overhead[i]))[259:403, 89:249]
        tile = cv2.resize(im, (174, 144))
        tile = cv2.copyMakeBorder(tile, 18,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(tile, episode.name[9:15]+"  "+str(i), (4,13), 0,.42,(235,235,235),1,cv2.LINE_AA)
        tiles.append(tile)
    return chosen, np.hstack(tiles)


if __name__ == "__main__":
    import json
    cv2.setNumThreads(1)
    base = Path(sys.argv[1]) / "recordings"
    print(json.dumps({name: review(base/name)[0] for name in sys.argv[2:]}))
