"""Inspect archive member names only; do not extract or publish payloads."""
import json,zipfile,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent
roots=[Path("C:/Users/Lenovo/Documents/ChatGPT/MLR"),Path("C:/Users/Lenovo/Downloads")]
rows=[]
for root in roots:
 for p in root.rglob("*"):
  if not p.is_file() or not p.name.lower().endswith((".zip",".tar",".tar.gz",".tgz")):continue
  if root.name=="Downloads" and not any(s in p.name.lower() for s in ("libero","mlr","demo","data","rctr")):continue
  row=dict(path=str(p),bytes=p.stat().st_size)
  try:
   if zipfile.is_zipfile(p):
    with zipfile.ZipFile(p) as z:names=z.namelist()
   elif p.stat().st_size<128*1024**2:
    with tarfile.open(p) as z:names=z.getnames()
   else:
    row["skipped"]="Compressed tar >128MiB; not inspected";rows.append(row);continue
   row["member_count"]=len(names)
   row["possible_raw_members"]=[n for n in names if n.lower().endswith((".hdf5",".h5")) or
     (Path(n).name.startswith("state_") and n.endswith(".npz"))]
  except Exception as e:row["error"]=repr(e)
  rows.append(row)
(R/"LOCAL_ARCHIVE_INVENTORY.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
print(json.dumps(dict(archives=len(rows),matches=[x for x in rows if x.get("possible_raw_members")],
                     skipped=[x for x in rows if "skipped" in x or "error" in x])))
