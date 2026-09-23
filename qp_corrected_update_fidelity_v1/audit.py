"""Independent arithmetic audit from raw CSV, without production summarizers."""
import numpy as np,pandas as pd
from .common import *
def run():
 starts=read(OUT/'START_MANIFEST.json')['starts'];stats=pd.read_csv(OUT/'START_LEVEL_METRICS.csv').set_index('start_id');checks=[]
 for s in starts:
  sid=s['start_id'];path=OUT/'starts'/sid;rec=read(path/'COMPLETE.json');old=pd.read_csv(path/'OLD.csv');new=pd.read_csv(path/'NEW.csv');row=stats.loc[sid]
  if rec['status']=='PASS':
   a=old.measured_delta_d0.to_numpy();b=old.measured_delta_dfull.to_numpy();tau=old.tau_zero.to_numpy();p=float(np.maximum(b,0).sum()/len(b)-np.maximum(a,0).sum()/len(a));assert np.isclose(p,row.damage_difference,atol=1e-12,rtol=1e-12)
   assert int(((a>tau)&(b<=tau)).sum())==row.prevented;assert int(((a<=tau)&(b>tau)).sum())==row.introduced
   assert np.isclose(float(new.measured_delta_dfull.mean()-new.measured_delta_d0.mean()),row.plasticity_cost,atol=1e-12,rtol=1e-12)
   assert rec['kkt_after']<=1e-8 and rec['projection_residual']<=5e-6 and rec['actual_positive']==0 and rec['source_restored']
  else:assert old.measured_delta_dfull.isna().all()
  checks.append(dict(start_id=sid,solver_status=rec['status'],arithmetic_and_accounting=True))
 freeze=read(OUT/'EXECUTION_FREEZE.json');unchanged={p:sha(ROOT/p)==h for p,h in freeze['files'].items()};assert all(unchanged.values())
 write(OUT/'INDEPENDENT_AUDIT.json',dict(status='PASS',starts=len(checks),checks=checks,frozen_sources_unchanged=unchanged,formal_training_resumed=False))
if __name__=='__main__':run()
