"""Publish a design-only report and provenance under the user's standing rule."""
from pathlib import Path
import hashlib,json,re,subprocess,datetime
R=Path(__file__).resolve().parent;repo=R.parents[1]/"publish_mlr_branch"
dest=repo/"reports/ra_epr_design_20261020"
def git(*args):return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()
assert git("branch","--show-current")=="MLR"
assert git("remote","get-url","origin")=="https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git"
assert not git("diff","--cached","--name-only"),"Preserve unrelated staged changes"
assert json.loads((R/"RUN_STATUS.json").read_text(encoding="utf8"))["status"]=="DESIGN_ONLY"
secret=re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}")
manifest={}
for name in ["DESIGN.md","audit_sources.py","SOURCE_AUDIT.json","RUN_STATUS.json","publish.py"]:
 p=R/name;raw=p.read_bytes();text=raw.decode("utf8")
 assert len(raw)<1024**2 and not secret.search(text)
 assert not re.search(r"password\s*=\s*[\x22\x27][^\x22\x27]+[\x22\x27]",text,re.I)
 dest.mkdir(parents=True,exist_ok=True);(dest/name).write_bytes(raw)
 manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
(dest/".gitattributes").write_bytes(b"* -text -whitespace\n")
(dest/"PUBLICATION_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
paths=[(dest/n).relative_to(repo).as_posix() for n in list(manifest)+[".gitattributes","PUBLICATION_MANIFEST.json"]]
subprocess.run(["git","-C",str(repo),"add","-f","--pathspec-from-file=-","--pathspec-file-nul"],input=b"\0".join(x.encode("utf8") for x in paths)+b"\0",check=True)
assert set(git("diff","--cached","--name-only").splitlines())<=set(paths)
subprocess.run(["git","-C",str(repo),"diff","--cached","--check"],check=True)
if git("diff","--cached","--name-only"):
 subprocess.run(["git","-C",str(repo),"commit","-q","-m","Design expert-validated executable policy replay complementary to RA"],check=True)
 subprocess.run(["git","-C",str(repo),"push","origin","MLR"],check=True)
head=git("rev-parse","HEAD");remote=git("ls-remote","origin","refs/heads/MLR").split()[0];assert head==remote
receipt=dict(status="PUSH_VERIFIED",experiment_status="DESIGN_ONLY",commit=head,remote=remote,files=len(paths),time=datetime.datetime.now().astimezone().isoformat())
(R/("PUBLICATION_RECEIPT_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json")).write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
