import tarfile,json,hashlib
from pathlib import Path
p=Path("C:/Users/Lenovo/Downloads/data.tar.gz")
names=[];error=None
try:
 with tarfile.open(p,"r|gz") as z:
  for m in z:names.append(m.name)
except Exception as e:error=repr(e)
o=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
       names_read=len(names),error=error,first30=names[:30],
       hdf5=[n for n in names if n.endswith((".hdf5",".h5"))],
       raw_npz=[n for n in names if Path(n).name.startswith("state_") and n.endswith(".npz")])
(Path(__file__).parent/"DATA_ARCHIVE_INSPECTION.json").write_text(json.dumps(o,indent=2)+"\n",encoding="utf-8")
print(json.dumps(o,indent=2))
