"""Read-only inventory for the new two-task experiment; no model or simulator."""
import json, os, subprocess, hashlib
from pathlib import Path
import h5py
P=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019')
S=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source')
out={'config':json.loads((P/'resolved_eval.json').read_text()),'training_config':json.loads((P/'resolved_ra_coreset.json').read_text()),'tasks':{},'sources':{}}
for t in (6,7):
 b=json.loads((P/'bindings'/('task%d.json'%t)).read_text()); ds=sorted({w['demo_id'] for w in b['windows'].values() if w['identity_unique']},key=lambda x:int(x.split('_')[-1]))
 with h5py.File(b['hdf5'],'r') as h:
  g=h['data'][ds[0]]; xml=g.attrs['model_file']; xml=xml.decode() if isinstance(xml,bytes) else xml
  out['tasks'][str(t)]={'demos':ds,'hdf5':b['hdf5'],'windows':len(b['windows']),'first_binding':next(iter(b['windows'].items())),'obs_shapes':{k:list(v.shape) for k,v in g['obs'].items()},'xml':xml,'root_attrs':{k:str(v) for k,v in h['data'].attrs.items()}}
for n in ('libero/lifelong/metric.py','libero/lifelong/models/modules/rgb_modules.py','libero/lifelong/models/modules/language_modules.py','libero/lifelong/models/modules/transformer_modules.py','coreset_exp/evaluate.py','coreset_exp/common.py','coreset_exp/model.py'):
 if (S/n).is_file():out['sources'][n]=(S/n).read_text()
out['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
out['processes']=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)
out['disk']={m:{'available':(s:=os.statvfs(m)).f_bavail*s.f_frsize,'free':s.f_bfree*s.f_frsize} for m in ('/data','/')}
out.update(server_writes=0,model_forwards=0,environment_steps=0)
print(json.dumps(out))
