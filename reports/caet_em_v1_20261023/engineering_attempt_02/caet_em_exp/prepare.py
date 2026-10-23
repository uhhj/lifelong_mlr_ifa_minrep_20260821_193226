"""Frozen assets and storage bounds; no training or environments."""
from .common import *
def main():
 contract=verify();manifest={};source_count=0
 for task in range(6,10):
  binding=js(PARENT/'bindings'/('task%d.json'%task));assert sha(binding['hdf5'])==binding['hdf5_sha256'];current=[]
  for name,w in binding['windows'].items():
   p=PARENT/'features'/str(task)/name;assert sha(p)==w['sha256'];current.append(dict(name=name,bytes=p.stat().st_size,sha256=w['sha256']))
  selection=js(PARENT/'ra_coreset/storage'/('task%d.json'%task))['selected'];assert len(selection)==QUOTAS[task]
  assert {r['destination']:r['sha256'] for r in selection}=={p.name:sha(p) for p in (PARENT/'ra_coreset/buffer').glob('t%d__*.pt'%task)}
  manifest[str(task)]=dict(current=current,hdf5_sha256=binding['hdf5_sha256'],binding_sha256=sha(PARENT/'bindings'/('task%d.json'%task)),selected=selection)
 # Concrete conservative bound: per arm <=1.8 GiB (4 full policy CP, full optimizer recovery,
 # atomic recovery temporary, exact original RA payloads, sidecars, teacher caches and outcomes).
 from .model import Codec,Policy
 stats={k+s:torch.zeros(d) if s=='_mean' else torch.ones(d) for k,d in [('action',7),('response',9),('visual',64)] for s in ('_mean','_std')}
 codec=Codec(stats);cs=codec.state_dict();estimated={};buffer_bytes=sum(Path(r['source']).stat().st_size for rows in manifest.values() for r in rows['selected'])
 for arm in ARMS:
  p=Policy(arm,cs,dict(mean=torch.zeros(64),std=torch.ones(64)));state_bytes=sum(v.numel()*v.element_size() for v in p.state_dict().values())
  estimated[arm]=dict(parameters=sum(v.numel() for v in p.parameters()),policy_bytes=state_bytes,buffer_bytes=buffer_bytes,bound_bytes=12*state_bytes+buffer_bytes+200*1024**2)
  assert estimated[arm]['bound_bytes']<=int(1.8*1024**3),estimated[arm]
 disk=capacity();data_need=int(2.1*1024**3);system_need=int(3.6*1024**3)
 assert disk['/data']['root_free']>=data_need+2*1024**3 and disk['/']['ordinary_free']>=system_need+1024**3,disk
 write(ROOT/'INPUT_MANIFEST.json',dict(status='PASS',contract=contract,tasks=manifest,official_sha256=sha(config().pretrain),population=POPS,quotas=QUOTAS))
 write(ROOT/'CAPACITY_PLAN.json',dict(status='PASS_NOT_FINAL',disk=disk,actual_tensor_estimates=estimated,data_worst_new=data_need,system_worst_new=system_need,reserve_data=2*1024**3,reserve_system=1024**3,per_arm_bound=int(1.8*1024**3),shared_bound=int(.3*1024**3)))
 event('prepare_complete',contract=contract)
if __name__=='__main__':main()
