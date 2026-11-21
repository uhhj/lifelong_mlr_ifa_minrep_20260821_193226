"""B1 image pairs and inference inputs, indexed on the host monotonic clock."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np


def save_pair(root, stamp, pair, pixels):
    if shutil.disk_usage(root).free < 128 * 1024**2:
        raise OSError("同步图像目录可用空间不足 128 MiB")
    folder = root / str(stamp // 60_000_000_000)
    folder.mkdir(exist_ok=True)
    scene = np.concatenate([f["image"] for f in pair], axis=1)
    model = cv2.cvtColor(np.concatenate(pixels, axis=1), cv2.COLOR_RGB2BGR)
    paths = {}
    for role, suffix, image, options in (
        ("model", ".png", model, [cv2.IMWRITE_PNG_COMPRESSION, 1]),
        ("scene", ".jpg", scene, [cv2.IMWRITE_JPEG_QUALITY, 92]),
    ):
        path = folder / (str(stamp) + suffix)
        if not cv2.imwrite(str(path), image, options):
            raise OSError(f"同步图像保存失败：{path.name}")
        paths[role] = path.relative_to(root).as_posix()
    return paths


class FrameArchive:
    def __init__(self, directory):
        self.root = Path(directory)
        self.history = deque()

    def __enter__(self):
        self.root.mkdir(mode=0o700)
        with ExitStack() as stack:
            self.index = stack.enter_context((self.root/"inputs.jsonl").open("x", encoding="utf8", buffering=1))
            self.writer = stack.enter_context(ThreadPoolExecutor(max_workers=1))
            self.stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self.stack.close()

    def predict(self, model, pair, pixels, query, stamp):
        # One pair can encode concurrently with its own inference. No backlog can grow.
        pending = self.writer.submit(save_pair, self.root, stamp, pair, pixels)
        pred = model.predict_pixels(pixels, query["q"])
        predicted_ns = time.monotonic_ns()
        paths = pending.result()
        self.history.append(stamp)
        while len(self.history) > len(model.history):
            self.history.popleft()
        history = [self.history[0]] * (model.history.maxlen-len(self.history)) + list(self.history)
        cameras = {role: {k: f[k] for k in ("sequence","host_monotonic_ns","read_started_monotonic_ns")}
                   for role,f in zip(("external","wrist"),pair)}
        row = dict(stamp=stamp, cameras=cameras, images=paths, history=history,
                   skew_ns=abs(pair[0]["host_monotonic_ns"]-pair[1]["host_monotonic_ns"]),
                   feedback_ns=query["feedback_ns"], q=query["q"], p=pred,
                   mode_at_query=query["mode_at_query"], predicted_ns=predicted_ns)
        try:
            self.index.write(json.dumps(row,allow_nan=False)+"\n")
        except OSError as exc:
            raise OSError(f"B1 同步记录写入失败：{exc}") from exc
        return pred, paths
