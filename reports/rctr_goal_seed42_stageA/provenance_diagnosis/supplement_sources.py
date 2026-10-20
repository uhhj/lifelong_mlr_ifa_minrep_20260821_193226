import json,urllib.request,hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
out={}
urls={
 "initial_requirements":"https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/4f0dd188960c25b608babc7de34e1e20a9d1306f/requirements.txt",
 "mujoco235":"https://pypi.org/pypi/mujoco/2.3.5/json",
 "issue16":"https://api.github.com/repos/Lifelong-Robot-Learning/LIBERO/issues/16/comments",
}
for k,u in urls.items():
 raw=urllib.request.urlopen(u,timeout=40).read()
 item=dict(url=u,sha256=hashlib.sha256(raw).hexdigest())
 if k=="mujoco235":
  d=json.loads(raw)
  item["wheels"]=[{a:x[a] for a in ("filename","digests","upload_time_iso_8601")} for x in d["urls"]
                  if "cp38" in x["filename"] and "manylinux" in x["filename"] and "x86_64.whl" in x["filename"]]
 elif k=="issue16":
  item["maintainer_comments"]=[dict(url=x["html_url"],date=x["created_at"],body=x["body"],association=x["author_association"])
    for x in json.loads(raw) if x["author_association"] in ("COLLABORATOR","MEMBER","OWNER")]
 else:item["text"]=raw.decode()
 out[k]=item
root="C:/Users/Lenovo/Documents/ChatGPT/MLR"
proc=subprocess.run(["rg","--files",root,"-g","*.hdf5","-g","*.h5","-g","*states*.npz","-g","*demo*.npz","-g","*libero*.zip"],
                    text=True,encoding="utf-8",capture_output=True)
out["local_search"]=dict(root=root,exit_code=proc.returncode,matches=proc.stdout.splitlines(),
                        interpretation="All matches are later experimental state traces, not original GOAL demonstration assets.")
(R/"SUPPLEMENTAL_SOURCES.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:list(v) for k,v in out.items()}))
