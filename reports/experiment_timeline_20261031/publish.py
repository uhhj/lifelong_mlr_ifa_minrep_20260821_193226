"""Publish only the exact reviewed historical timeline and source hashes."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
REPO = WORK / "publish_mlr_branch"
DEST = REPO / "reports/experiment_timeline_20261031"
URL = "https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git"
NAMES = ['TIMELINE.md', 'FULL_TIMELINE.md', 'TIMELINE.csv', 'TIMELINE.json', 'SOURCE_AUDIT.json', 'HISTORICAL_COMMIT.json', 'part_early.json', 'part_align.json', 'part_shbr.json', 'part_late.json', 'build_timeline.py', 'publish.py']


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), "-c", "http.sslBackend=openssl",
                                    "-c", "http.version=HTTP/1.1", *args], text=True, encoding="utf-8").strip()


def main():
    assert git("branch", "--show-current") == "MLR"
    assert git("remote", "get-url", "origin") == URL
    assert not git("diff", "--cached", "--name-only"), "Preserve unrelated staged changes"
    audit = json.loads((HERE / "SOURCE_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS_TIMELINE_SOURCE_BINDING_AND_DIRECTORY_COVERAGE"
    for rel, expected in audit["source_bindings"].items():
        raw = (WORK / rel).read_bytes()
        assert len(raw) == expected["bytes"] and hashlib.sha256(raw).hexdigest() == expected["sha256"], rel
    manifest = {}
    for name in NAMES:
        src = HERE / name
        assert src.is_file() and not src.is_symlink() and src.suffix in {".md", ".json", ".py", ".csv"}
        raw = src.read_bytes()
        assert 0 < len(raw) < 500000
        text = raw.decode("utf-8-sig")
        assert not re.search(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}", text)
        assert not re.search(r"(?:password|passwd|secret|token)\s*[:=]\s*['\"][^'\"]{6,}['\"]", text, re.I)
        if src.suffix == ".json":
            json.loads(text)
        dst = DEST / name
        if dst.exists():
            assert dst.read_bytes() == raw, "Do not overwrite unrelated publication edits"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == digest
        manifest[name] = {"bytes": len(raw), "sha256": digest}
    (DEST / "PUBLICATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    (DEST / ".gitattributes").write_bytes(b"* -text -whitespace\n")
    names = NAMES + ["PUBLICATION_MANIFEST.json", ".gitattributes"]
    paths = [(DEST / n).relative_to(REPO).as_posix() for n in names]
    subprocess.run(["git", "-C", str(REPO), "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul"],
                   input=b"\0".join(x.encode() for x in paths)+b"\0", check=True)
    assert set(git("diff", "--cached", "--name-only").splitlines()) <= set(paths)
    git("diff", "--cached", "--check")
    for path in paths:
        staged = subprocess.check_output(["git", "-C", str(REPO), "show", ":"+path])
        assert staged == (REPO/path).read_bytes(), "Staged content differs: "+path
    changed = bool(git("diff", "--cached", "--name-only"))
    if changed:
        git("commit", "-m", "Document source-bound experiment timeline from reproduction to SHBR")
    head = git("rev-parse", "HEAD")
    remote = git("ls-remote", "origin", "refs/heads/MLR").split()[0]
    if remote != head:
        git("push", "origin", "MLR")
    verified = git("ls-remote", "origin", "refs/heads/MLR").split()[0]
    assert head == verified
    receipt = {"status": "PUSH_VERIFIED", "commit": head, "remote_sha": verified, "remote": URL,
               "branch": "MLR", "files": len(paths), "new_commit": changed,
               "observed_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
               "scope": "Local read-only evidence analysis; no new experiment or acceptance change",
               "payload_published": False}
    (HERE / "PUBLICATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
