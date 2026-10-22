"""Publish reviewed implementation/metadata only, never downloaded experiment assets."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess

R = Path(__file__).resolve().parent
REPO = R.parents[1] / "publish_mlr_branch"
DEST = REPO / "reports/align_object_extension_goal_conditional"


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def inspect(value):
    if isinstance(value, dict):
        assert not {"password", "private_key", "last_action", "latent_tensor"} & set(value)
        for v in value.values():
            inspect(v)
    elif isinstance(value, list):
        for v in value:
            inspect(v)


def main():
    assert git("branch", "--show-current") == "MLR"
    assert git("remote", "get-url", "origin") == "https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git"
    assert not git("diff", "--cached", "--name-only"), "Unrelated staged edits"
    allowed = {".py", ".json", ".jsonl", ".md", ".yaml", ".yml", ".sh", ".csv", ".exit", ".txt"}
    secret = re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}")
    exclude = {"STOP_LIGHT_BUNDLE.json", "PARENT_SOURCE_BUNDLE.json", "DEPLOY_TEXT_BUNDLE.json"}
    manifest = {}
    for p in sorted(R.rglob("*")):
        if (not p.is_file() or p.suffix not in allowed or p.name in exclude or
                ("BUNDLE" in p.name and p.suffix == ".json") or
                p.name.startswith("PUBLICATION_RECEIPT") or
                {"__pycache__", "asset_sources"} & set(p.relative_to(R).parts)):
            continue
        raw = p.read_bytes()
        assert len(raw) < 8*1024**2, str(p)
        txt = raw.decode("utf-8-sig")
        assert not secret.search(txt), str(p)
        assert not re.search(r"password\s*=\s*['\"][^'\"]+['\"]", txt, re.I), str(p)
        if p.suffix == ".json":
            inspect(json.loads(txt))
        if p.suffix == ".jsonl":
            for line in txt.splitlines():
                inspect(json.loads(line))
        rel = p.relative_to(R).as_posix()
        out = DEST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)
        manifest[rel] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (DEST/".gitattributes").write_bytes(b"* -text -whitespace\n")
    (DEST/"PUBLICATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n")
    paths = [(DEST/n).relative_to(REPO).as_posix()
             for n in [*manifest, ".gitattributes", "PUBLICATION_MANIFEST.json"]]
    subprocess.run(["git", "-C", str(REPO), "add", "-f", "--pathspec-from-file=-",
                    "--pathspec-file-nul"], input=b"\0".join(x.encode() for x in paths)+b"\0", check=True)
    assert set(git("diff", "--cached", "--name-only").splitlines()) <= set(paths)
    subprocess.run(["git", "-C", str(REPO), "diff", "--cached", "--check"], check=True)
    if git("diff", "--cached", "--name-only"):
        subprocess.run(["git", "-C", str(REPO), "commit", "-q", "-m",
                        "Record OBJECT42 Task8 progress with frozen ALIGN checks"], check=True)
        subprocess.run(["git", "-C", str(REPO), "push", "origin", "MLR"], check=True)
    head = git("rev-parse", "HEAD")
    remote = git("ls-remote", "origin", "refs/heads/MLR").split()[0]
    assert head == remote
    receipt = dict(status="PUSH_VERIFIED", commit=head, remote=remote,
                   time=datetime.datetime.now().astimezone().isoformat(),
                   files=len(paths), bytes=sum(x["bytes"] for x in manifest.values()),
                   experiment_status=json.loads((R/"RUN_STATUS.json").read_text(encoding="utf8"))["status"])
    (R/("PUBLICATION_RECEIPT_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json")).write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
