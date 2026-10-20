"""External path-only adapter. Preserve first failed log/exit and frozen probe."""
import json,hashlib
from pathlib import Path
import h5py
import state_probe as p
original=p.O
assert not (original/"recovery_asset_paths").exists()
c=json.loads((original/"RESTORATION_CONTRACT.json").read_text())
for name,h in c["files"].items():assert p.sha(name)==h,name
events=[json.loads(x) for x in (original/"state_attempts.jsonl").read_text().splitlines()]
assert not any(x["event"]=="engineering_step_start" for x in events)
assert events[-1]["event"]=="environment_closed"
assert (p.W/"probe.exit").read_text().strip()=="1"
old=p.relocate
def adapted(xml):
 root=p.ET.fromstring(xml);mapped=[]
 for node in root.iter():
  path=node.get("file")
  if path and "/chiliocosm/assets/" in path:
   suffix=path.split("/chiliocosm/assets/",1)[1]
   new=(p.S/"libero"/"libero"/"assets"/suffix).resolve()
   assert new.is_file(),str(new)
   node.set("file",str(new));mapped.append(dict(old=path,new=str(new),sha256=p.sha(new)))
 resolved,other=old(p.ET.tostring(root,encoding="unicode"))
 return resolved,mapped+other
p.relocate=adapted
a=json.loads((p.W/"ASSETS.json").read_text())
with h5py.File(a["hdf5"],"r") as f:
 records=[]
 for row in a["chosen"]:
  xml,changes=adapted(f["data"][row["demo_id"]].attrs["model_file"])
  records.append(dict(demo_id=row["demo_id"],changes=changes,xml_sha256=hashlib.sha256(xml.encode()).hexdigest()))
p.write(p.W/"RESOURCE_RECOVERY.json",dict(reason="Original demonstration XML uses historical chiliocosm asset prefix",
 original_exit=1,original_engineering_steps=0,original_environment_constructions=1,
 original_inputs_and_source_unchanged=True,only_change="Resolve historical asset prefix to existing installed asset files; preserve all numerical XML attributes",records=records))
p.O=original/"recovery_asset_paths"
p.main()
