"""Read-only deployment inventory, no policy execution and no GPU tests."""
import json,time
from pathlib import Path
from flow_support import N,sha,write
from flow_metrics import metrics,pack
from prepare_zr import parent_spec
def main():
 rows={}
 for branch,seeds in [('GOAL_RA',[41,42,43]),('GOAL_Coreset',[41,42,43,44])]:
  for seed in seeds:
   sp=parent_spec(branch,seed);P=Path(sp['parent']);S=Path(sp['parent_source']);a=sp['parent_arm']
   assert json.loads(Path(sp['parent_engineering_file']).read_text())['status']=='PASS'
   assert Path(sp['parent_env']).is_file() and S.is_dir()
   m,r,records,h=metrics(P/a/'evaluation',seed,a)
   hashes={}
   for t in range(6,10):
    cp=P/a/'models'/('mlrifa_%d_model_ep99.pth'%t);hashes[str(t)]=sha(cp);assert hashes[str(t)]==records[t,t,0]['checkpoint_sha256']
    assert (P/a/'storage'/('task%d.json'%t)).is_file()
   rows[branch+str(seed)]=dict(parent=sp,metrics=pack(m),checkpoint_hashes=hashes,episode_hashes=h,source_contract_sha256=sha(P/'CONTRACT.json'))
 original=Path('/root/mlr_ifa/experiment_workspaces/ZRv2_OBJECT_seed43_20261018')
 for name in ['model.py','diagnostics.py']:
  assert sha(N/'zr_template/source/zr2_exp'/name)==sha(original/'source/zr2_exp'/name),name
 assert sha(N/'zr_template/attachment/reference_core.py')==sha(original/'attachment/reference_core.py')
 write(N/'PREDEPLOY_RESOURCES.json',dict(status='PASS_READONLY_NOT_GPU_ADMISSION',parents=rows,scientific_ZRv2_model_diagnostics_reference_core_bytes_same=True,optimizer_calls=0,environment_steps=0,time=time.time()))
 print('PASS: seven existing GOAL parents bound; ZRv2 numerical core byte-identical; zero GPU updates/environments')
if __name__=='__main__':main()
