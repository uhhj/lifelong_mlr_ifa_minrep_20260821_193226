from actual_update_fidelity_v1.common import read,write,sha,table,lines,emit,ROOT
from pathlib import Path
OUT=ROOT/'reports/qp_corrected_update_fidelity_v1'
SOURCE=ROOT/'reports/actual_update_fidelity_v1'
SCRATCH=Path('/root/qp_corrected_update_fidelity_v1')

def table(p,rows):
 """Heterogeneous success/failure records retain every field, atomically."""
 import csv,os
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with tmp.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
 os.replace(tmp,p)
