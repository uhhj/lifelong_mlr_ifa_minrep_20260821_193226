"""Actual full completed-stage GPU state roundtrip, without replaying training."""
from .common import *
from .model import Policy,strict_policy_restore
from .train import load_codec,optimizer
from .dataset import diagnostic_dataset
from .admission import equal
def main():
 contract=verify();capacity();proof=js(ROOT/'RESUME_MANIFEST.json');pack,codec=load_codec();seed(410010)
 p=Policy('B0',pack['model'],pack['statistics']).cuda();opt,sched,names=optimizer(p,22700)
 cp=arm_root('B0')/'models/task7.pth';raw=torch.load(cp,map_location='cpu');strict_policy_restore(p,raw,contract,sha(frozen_codec()))
 recovery=Path(proof['preserved_recovery']['path']);assert sha(recovery)==proof['preserved_recovery']['sha256']
 state=torch.load(recovery,map_location='cpu');assert state['task']==7 and state['next_epoch']==100 and state['steps']==38400 and state['cursor']==0
 assert state['contract']==proof['parent_contract'] and state['codec_sha256']==sha(frozen_codec())
 assert state['dataset_sha256']==sha(arm_root('B0')/'dataset/task7.json') and state['teacher_sha256'] is None
 assert equal(state['model'],raw['model'])
 restore(p,opt,sched,state);actual=capture(p,opt,sched)
 for key in ('model','optimizer','scheduler','rng','grads','modes','requires'):assert equal(actual[key],state[key]),key
 manifests={}
 for task in (6,7):
  ds=diagnostic_dataset('B0',task);manifests[str(task)]=dict(population=len(ds),manifest_sha256=sha(arm_root('B0')/'dataset'/('task%d.json'%task)),all_identity_hash_mask_order_exact=True)
 write(ROOT/'CONTINUATION_ADMISSION.json',dict(status='PASS',contract=contract,parent_contract=proof['parent_contract'],actual_GPU_full_state_roundtrip=True,restored_fields=['model','optimizer','scheduler','rng','grads','modes','requires','cursor','dataset_sha256','codec_sha256','teacher_sha256'],complete_epoch=100,completed_task=7,steps=38400,checkpoint_sha256=sha(cp),recovery_sha256=sha(recovery),diagnostic_population=manifests,optimizer_calls=0,forward_calls=0,environment_steps=0,retrain_completed=False))
 event('continuation_admission_complete',completed_task=7,steps=38400)
if __name__=='__main__':main()
