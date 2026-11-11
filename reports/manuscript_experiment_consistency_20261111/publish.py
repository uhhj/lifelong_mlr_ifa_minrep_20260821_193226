"""Publish only this lightweight audit under the standing workspace authorization."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
REPO = WS / "publish_mlr_branch"
DEST = REPO / "reports/manuscript_experiment_consistency_20261111"
FILES = ["REPORT.md", "AUDIT.json", "audit.py", "extract.py", "SOURCE_DOCUMENTS.json", "publish.py"]
ORIGIN = "https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git"

def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), "-c", "http.sslBackend=openssl", "-c", "http.version=HTTP/1.1", *args], text=True, encoding="utf8").strip()

assert git("branch", "--show-current") == "MLR"
assert git("remote", "get-url", "origin") == ORIGIN
assert not git("diff", "--cached", "--name-only"), "Unrelated staged changes"
assert not git("status", "--porcelain", "--", "reports/manuscript_experiment_consistency_20261111"), "Existing task destination changes"
audit = json.loads((ROOT / "AUDIT.json").read_text(encoding="utf8"))
assert audit["status"] == "PASS_NUMERIC_CROSSCHECK_WITH_TEXTUAL_CONFIGURATION_FINDINGS"
for rel, binding in audit["source_bindings"].items():
    raw = (WS / rel).read_bytes()
    assert len(raw) == binding["bytes"] and hashlib.sha256(raw).hexdigest() == binding["sha256"], rel

secret = re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}")
manifest = {}
reviewed = {}
for name in FILES:
    p = ROOT / name
    assert p.suffix in {".md", ".json", ".py"}
    raw = p.read_bytes()
    assert len(raw) < 2 * 1024 * 1024
    text = raw.decode("utf8")
    assert not secret.search(text), name
    if p.suffix == ".json":
        json.loads(text)
    if p.suffix == ".py":
        import ast
        ast.parse(text)
    reviewed[name] = raw
    manifest[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

DEST.mkdir(parents=True, exist_ok=True)
for name, raw in reviewed.items():
    (DEST / name).write_bytes(raw)
(DEST / "PUBLICATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
paths = [(DEST / name).relative_to(REPO).as_posix() for name in [*FILES, "PUBLICATION_MANIFEST.json"]]
subprocess.run(["git", "-C", str(REPO), "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul"], input=b"\0".join(p.encode("utf8") for p in paths) + b"\0", check=True)
assert set(git("diff", "--cached", "--name-only").splitlines()) <= set(paths)
git("-c", "core.whitespace=blank-at-eol,space-before-tab,cr-at-eol", "diff", "--cached", "--check")
if git("diff", "--cached", "--name-only"):
    git("commit", "-m", "Audit DEBR manuscript experiment configuration consistency")
git("push", "origin", "MLR")
head = git("rev-parse", "HEAD")
remote = git("ls-remote", "origin", "refs/heads/MLR").split()[0]
assert head == remote
receipt = {"status": "PUSH_VERIFIED", "commit": head, "remote_sha": remote, "branch": "MLR", "origin": ORIGIN,
    "files": len(paths), "bytes": sum(v["bytes"] for v in manifest.values()),
    "observed": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
(ROOT / "PUBLICATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
print(json.dumps(receipt, indent=2))
