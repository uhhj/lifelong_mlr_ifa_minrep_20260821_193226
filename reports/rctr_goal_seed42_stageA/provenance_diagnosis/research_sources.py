"""Fetch primary provenance metadata only; never download demonstration payload."""
import json,hashlib,urllib.request,datetime
from pathlib import Path
R=Path(__file__).resolve().parent
urls={
 "libero_requirements":"https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/requirements.txt",
 "collection":"https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/scripts/collect_demonstration.py",
 "conversion":"https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/scripts/create_dataset.py",
 "collection_wrapper":"https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.0/robosuite/wrappers/data_collection_wrapper.py",
 "robosuite_setup":"https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.0/setup.py",
 "hf_tree":"https://huggingface.co/api/datasets/yifengzhu-hf/LIBERO-datasets/tree/main/libero_goal?recursive=true",
 "hf_info":"https://huggingface.co/api/datasets/yifengzhu-hf/LIBERO-datasets",
 "collection_history":"https://api.github.com/repos/Lifelong-Robot-Learning/LIBERO/commits?path=scripts/collect_demonstration.py&per_page=100",
 "requirements_history":"https://api.github.com/repos/Lifelong-Robot-Learning/LIBERO/commits?path=requirements.txt&per_page=100",
 "mujoco237":"https://pypi.org/pypi/mujoco/2.3.7/json",
}
out={}
for name,url in urls.items():
 try:
  raw=urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"RCTR-provenance-audit"}),timeout=40).read()
  txt=raw.decode()
  item=dict(url=url,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
  if name=="hf_tree":item["entries"]=json.loads(txt)
  elif name=="hf_info":
   obj=json.loads(txt);item["metadata"]={k:obj.get(k) for k in ("id","sha","createdAt","lastModified","cardData")}
  elif name.endswith("history"):
   item["commits"]=[dict(sha=v["sha"],date=v["commit"]["committer"]["date"],message=v["commit"]["message"]) for v in json.loads(txt)]
  elif name=="mujoco237":
   obj=json.loads(txt);item["wheels"]=[{k:v[k] for k in ("filename","digests","upload_time_iso_8601","url")} for v in obj["urls"] if "cp38" in v["filename"] and "manylinux" in v["filename"]]
  else:
   terms=("mujoco","robosuite","states","state","ctrl","action","initial","version","date","time","env_info","cap_index","np.savez")
   item["relevant_lines"]=[dict(line=i+1,text=l) for i,l in enumerate(txt.splitlines()) if any(t in l for t in terms)]
  out[name]=item
 except Exception as e:out[name]=dict(url=url,error=repr(e))
(R/"PRIMARY_SOURCES.json").write_text(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),sources=out),indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:{"error":v.get("error"),"bytes":v.get("bytes")} for k,v in out.items()}))
