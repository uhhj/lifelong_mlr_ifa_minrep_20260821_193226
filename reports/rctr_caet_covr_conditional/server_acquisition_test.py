from pathlib import Path
import json,h5py,numpy as np
from acquire_libero50_server import repack
r=Path(__file__).resolve().parent/'packing_preflight'
r.mkdir(exist_ok=False)
a=r/'source.hdf5';b=r/'packed.hdf5'
with h5py.File(a,'w') as f:
 f.attrs['description']='immutable acquisition test'
 d=f.create_group('data/demo_0');d.attrs['model_file']='<mujoco/>'
 d.create_dataset('obs/rgb',data=np.random.RandomState(3).randint(0,256,(33,12,12,3),dtype=np.uint8))
 d.create_dataset('states',data=np.arange(33*110,dtype=np.float64).reshape(33,110))
 d.create_dataset('scalar',data=np.int64(17))
 d.create_dataset('empty',shape=(0,7),dtype=np.float32)
proof=repack(a,b)
assert proof['logical_datasets']==4 and proof['all_dataset_bytes_equal']
print(json.dumps(dict(status='PASS',proof=proof,environment_steps=0,optimizer_calls=0)))
