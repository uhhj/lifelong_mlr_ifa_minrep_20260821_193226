"""Read-only source snapshot, no model or data payloads."""
from pathlib import Path
import hashlib
import json

S = Path("/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019/seed42/source")
Q = Path("/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020")
files = {}
for p in S.rglob("*"):
    if (not p.is_file() or p.is_symlink() or "__pycache__" in p.parts or
            p.suffix not in {".py", ".yaml", ".yml", ".sh", ".sbatch", ".md", ".txt"}):
        continue
    raw = p.read_bytes()
    assert len(raw) < 2*1024**2
    files[p.relative_to(S).as_posix()] = dict(
        text=raw.decode("utf8"), sha256=hashlib.sha256(raw).hexdigest())
(Q/"PARENT_SOURCE_BUNDLE.json").write_text(json.dumps(files))
print(json.dumps(dict(files=len(files), bytes=sum(len(x["text"]) for x in files.values()))))
