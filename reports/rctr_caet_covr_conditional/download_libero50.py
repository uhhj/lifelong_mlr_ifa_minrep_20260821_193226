"""Download immutable official-mirror kitchen files; data stay outside publication.

The upstream LIBERO download_utils.py names this HF mirror. Hashes are LFS SHA256.
This downloads assets only, without choosing a training split or loading models.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import requests

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "local_assets" / "libero50_official"
REPO = "yifengzhu-hf/LIBERO-datasets"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4*1024**2), b""):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf8")
    os.replace(tmp, path)


def acquire(item, revision):
    path = DATA / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    status = path.with_suffix(".acquisition.json")
    expected = item["lfs"]["oid"]
    if path.exists():
        if path.stat().st_size == item["size"] and sha(path) == expected:
            return dict(path=item["path"], status="VERIFIED", sha256=expected,
                        bytes=item["size"], reused=True)
        raise RuntimeError("Existing final asset does not match: " + str(path))
    partial = path.with_suffix(".hdf5.partial")
    url = "https://huggingface.co/datasets/{}/resolve/{}/{}".format(
        REPO, revision, item["path"])
    failures = []
    for attempt in range(1, 9):
        start = partial.stat().st_size if partial.exists() else 0
        write(status, dict(status="DOWNLOADING", path=item["path"], start=start,
                           attempt=attempt, failures=failures, unix=time.time()))
        try:
            if start < item["size"]:
                # Signed redirects are never serialized or printed.
                with requests.get(url, headers={"Range": "bytes={}-".format(start)}
                                  if start else {}, stream=True,
                                  timeout=(30, 90)) as response:
                    response.raise_for_status()
                    append = start > 0 and response.status_code == 206
                    if append:
                        if not response.headers.get("Content-Range", "").startswith(
                                "bytes {}-".format(start)):
                            raise RuntimeError("Unexpected Content-Range")
                    with partial.open("ab" if append else "wb") as f:
                        last = time.monotonic()
                        for chunk in response.iter_content(4*1024**2):
                            if chunk:
                                f.write(chunk)
                            if time.monotonic()-last > 20:
                                f.flush()
                                write(status, dict(status="DOWNLOADING",
                                    path=item["path"], bytes=partial.stat().st_size,
                                    expected_bytes=item["size"], attempt=attempt,
                                    failures=failures, unix=time.time()))
                                last = time.monotonic()
            if partial.stat().st_size != item["size"]:
                raise RuntimeError("Downloaded size mismatch")
            got = sha(partial)
            if got != expected:
                # Retain corrupt bytes for diagnosis; never relabel them as valid.
                partial.rename(partial.with_suffix(".badsha-{}".format(int(time.time()))))
                raise RuntimeError("Downloaded SHA256 mismatch")
            os.replace(partial, path)
            result = dict(status="VERIFIED", path=item["path"], sha256=got,
                          bytes=path.stat().st_size, unix=time.time(), failures=failures)
            write(status, result)
            return result
        except Exception as error:
            # Do not serialize exception URLs, which may include signed redirects.
            failures.append(dict(attempt=attempt, error_type=type(error).__name__,
                                 unix=time.time()))
            write(status, dict(status="RETRYING", path=item["path"], failures=failures))
            time.sleep(min(5*attempt, 40))
    raise RuntimeError("Download retries exhausted for " + item["path"])


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = HERE / "LIBERO50_DOWNLOAD_MANIFEST.json"
    if manifest.exists():
        v = json.loads(manifest.read_text(encoding="utf8"))
    else:
        info = requests.get("https://huggingface.co/api/datasets/" + REPO, timeout=30)
        info.raise_for_status()
        revision = info.json()["sha"]
        tree = requests.get("https://huggingface.co/api/datasets/{}/tree/{}?recursive=true&limit=1000".format(REPO, revision), timeout=45)
        tree.raise_for_status()
        items = [x for x in tree.json() if x["type"] == "file" and
                 x["path"].startswith(("libero_90/KITCHEN", "libero_10/KITCHEN"))
                 and x["path"].endswith(".hdf5")]
        assert len(items) == 50 and all(len(x["lfs"]["oid"]) == 64 for x in items)
        v = dict(repo=REPO, revision=revision, files=items,
                 scope="All 50 kitchen files; no pretrained checkpoint supplied",
                 upstream="https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/utils/download_utils.py")
        write(manifest, v)
    remaining = sum(x["size"] for x in v["files"]
                    if not (DATA/x["path"]).exists())
    assert shutil.disk_usage(DATA).free > remaining + 8*1024**3, "Insufficient local space"
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(acquire, x, v["revision"]): x for x in v["files"]}
        for future in concurrent.futures.as_completed(pending):
            item = pending[future]
            try:
                results.append(future.result())
            except Exception as error:
                results.append(dict(path=item["path"], status="FAILED",
                                    error_type=type(error).__name__))
            write(HERE/"LIBERO50_DOWNLOAD_STATUS.json", dict(
                status="DOWNLOADING", unix=time.time(), finished=len(results),
                total=50, results=results, data_directory=str(DATA),
                pretrained_checkpoint="NOT_FOUND_IN_AUTHOR_RELEASE"))
    ok = len(results) == 50 and all(x["status"] == "VERIFIED" for x in results)
    write(HERE/"LIBERO50_DOWNLOAD_STATUS.json", dict(
        status="DATA_VERIFIED" if ok else "DOWNLOAD_INCOMPLETE", unix=time.time(),
        finished=len(results), total=50, results=results,
        pretrained_checkpoint="NOT_FOUND_IN_AUTHOR_RELEASE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
