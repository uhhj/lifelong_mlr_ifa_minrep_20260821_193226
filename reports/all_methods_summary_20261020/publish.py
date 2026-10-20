"""Publish only reviewed consolidation code, report, metadata and generated workbook."""
from pathlib import Path
import json,hashlib,re,subprocess,datetime,zipfile
R=Path(__file__).resolve().parent;repo=R.parents[1]/"publish_mlr_branch"
dest=repo/"reports/all_methods_summary_20261020"
def git(*args):
    return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()
assert git("branch","--show-current")=="MLR"
assert git("remote","get-url","origin")=="https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git"
assert not git("diff","--cached","--name-only"),"Preserve unrelated staged changes"
assert json.loads((R/"RUN_STATUS.json").read_text(encoding="utf8"))["status"]=="COMPLETED_READONLY_CONSOLIDATION"
secret=re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}")
def inspect(v):
    if isinstance(v,dict):
        assert not {"password","private_key","last_action","latent_tensor"}&set(v)
        for z in v.values():inspect(z)
    elif isinstance(v,list):
        for z in v:inspect(z)
manifest={}
for p in R.rglob("*"):
    if not p.is_file() or "__pycache__" in p.parts or p.name.startswith("PUBLICATION_RECEIPT"):continue
    assert p.suffix in {".py",".mjs",".json",".md",".csv",".xlsx"},p.name
    raw=p.read_bytes();assert len(raw)<8*1024**2,p.name
    if p.suffix==".xlsx":
        assert p.parent.name=="deliverables"
        with zipfile.ZipFile(p) as z:
            assert not any("vba" in n.lower() or "externallinks" in n.lower() for n in z.namelist())
            for n in z.namelist():
                assert n.endswith((".xml",".rels")),n
                assert not secret.search(z.read(n).decode("utf8")),n
    else:
        text=raw.decode("utf-8-sig");assert not secret.search(text),p.name
        assert not re.search(r"password\s*=\s*[\x22\x27][^\x22\x27]+[\x22\x27]",text,re.I),p.name
        if p.suffix==".json":inspect(json.loads(text))
    rel=p.relative_to(R).as_posix()
    q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw)
    manifest[rel]={"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
(dest/".gitattributes").write_bytes(b"* -text -whitespace\n")
(dest/"PUBLICATION_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")
paths=[(dest/n).relative_to(repo).as_posix() for n in list(manifest)+[".gitattributes","PUBLICATION_MANIFEST.json"]]
subprocess.run(["git","-C",str(repo),"add","-f","--pathspec-from-file=-","--pathspec-file-nul"],
    input=b"\0".join(x.encode("utf8") for x in paths)+b"\0",check=True)
# NUL output keeps the workbook's Unicode filename unquoted.
staged=subprocess.check_output(["git","-C",str(repo),"diff","--cached","--name-only","-z"]).decode("utf8").split("\0")
assert set(filter(None,staged))<=set(paths)
subprocess.run(["git","-C",str(repo),"diff","--cached","--check"],check=True)
if any(staged):
    subprocess.run(["git","-C",str(repo),"commit","-q","-m","Consolidate all Coreset-line experiments and metrics into audited workbook"],check=True)
    subprocess.run(["git","-C",str(repo),"push","origin","MLR"],check=True)
head=git("rev-parse","HEAD");remote=git("ls-remote","origin","refs/heads/MLR").split()[0]
assert head==remote
receipt={"status":"PUSH_VERIFIED","commit":head,"remote":remote,"files":len(paths),
    "bytes":sum(x["bytes"] for x in manifest.values()),"time":datetime.datetime.now().astimezone().isoformat()}
(R/("PUBLICATION_RECEIPT_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json")).write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
