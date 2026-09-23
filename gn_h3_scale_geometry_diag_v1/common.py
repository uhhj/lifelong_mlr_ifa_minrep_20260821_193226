from gn_functional_geometry_v1.common import ROOT,sha,read,write,save
from pathlib import Path
import csv,os
OUT=ROOT/'reports/gn_h3_scale_geometry_diag_v1'
OLD=ROOT/'reports/gn_functional_geometry_v1'
def table(name,rows):
 p=OUT/name;t=p.with_name(p.name+'.tmp')
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with t.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
 os.replace(t,p)
