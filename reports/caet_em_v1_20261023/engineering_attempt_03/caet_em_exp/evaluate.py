"""Full strict restore, original evaluator and RNG ordering, first of four actions."""
import argparse
from torch import nn
from .common import *
from .model import Policy,strict_policy_restore,selected
class Deployed(nn.Module):
 def __init__(self,frontend,policy):super().__init__();self.frontend=frontend;self.policy=policy
 def reset(self):pass # evaluator owns the sole 1..8 observation history
 @torch.no_grad()
 def get_action(self,data):
  self.eval();native=self.frontend.spatial_encode(self.frontend.preprocess_input(data,train_mode=False));q,_,_=self.policy(native);m,s,_=selected(q)
  # Sample ONLY the executed first action. Sampling unused steps would change global RNG.
  action=torch.distributions.Normal(m[:,0],s[:,0]).sample();return action.cpu().numpy()
class Algorithm(nn.Module):
 def __init__(self,p):super().__init__();self.policy=p
 def reset(self):self.policy.reset()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('arm',choices=ARMS);args=ap.parse_args();contract=verify();root=arm_root(args.arm);assert js(root/'TRAINING_COMPLETE.json')['steps']==105500
 out=root/'evaluation';out.mkdir(exist_ok=True);assert not list(out.glob('trials*.jsonl')) and not list(out.glob('attempts*.jsonl')),'No blind partial evaluation restart'
 from coreset_exp.model import data_setup
 from coreset_exp import evaluate as context
 from coreset_exp.eval_metric import evaluate_one_task_success
 from libero.lifelong.utils import control_seed
 c=config();c.experiment_dir=str(out);assert c.eval.n_eval==20 and c.eval.max_steps==600;control_seed(41);benchmark,ds,shapes=data_setup(c,True);c.shape_meta=shapes;torch.backends.cudnn.benchmark=True
 pack=torch.load(frozen_codec(),map_location='cpu');loads=[]
 for stage in range(6,10):
  control_seed(41)
  for taskid in range(6,stage+1):
   checkpoint=root/'models'/('task%d.pth'%stage);raw=torch.load(checkpoint,map_location='cpu');frontend=full_frontend();saved=rng()
   try:
    p=Policy(args.arm,pack['model'],pack['statistics']).cuda();n=strict_policy_restore(p,raw,contract,sha(frozen_codec()));p.eval();algo=Algorithm(Deployed(frontend,p)).cuda().eval()
   finally:restore_rng(saved)
   task=benchmark.get_task(taskid);ip=Path(c.init_states_folder)/task.problem_folder/task.init_states_file;init=torch.load(ip);assert len(init)>=20
   ctx=dict(arm=args.arm,stage=stage,epoch=99,checkpoint_sha256=sha(checkpoint),codec_sha256=sha(frozen_codec()),contract_sha256=contract,loaded_downstream_keys=n,initial_file_sha256=sha(ip),initial_hashes=[hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for v in init[:20]],trial_log=str(out/('trials_stage%d_task%d.jsonl'%(stage,taskid))),attempt_log=str(out/('attempts_stage%d_task%d.jsonl'%(stage,taskid))))
   context.CONTEXT.clear();context.CONTEXT.update(ctx);save(out/('rng_before_stage%d_task%d.pth'%(stage,taskid)),rng());loads.append(dict(stage=stage,task=taskid,strict_keys=n,policy_sha256=sha(checkpoint),codec_sha256=sha(frozen_codec()),single_history=True,executed_future_steps=1))
   write(root/'CHECKPOINT_LOAD_AUDIT.json',loads)
   try:
    instr,mask=benchmark.get_task_emb(taskid);score=evaluate_one_task_success(c,algo,task,instr,mask,taskid,see=41);write(out/('cell_stage%d_task%d.json'%(stage,taskid)),dict(success_rate=float(score),**ctx))
   except BaseException as e:context.ledger('technical_exception',exception=repr(e));raise
   finally:del algo,p,frontend;torch.cuda.empty_cache()
   event('cell_complete',arm=args.arm,stage=stage,task=taskid)
 write(out/'EVALUATION_COMPLETE.json',dict(status='PASS',episodes=200,vector_attempts=20,cells=10,contract=contract));event('evaluation_complete',arm=args.arm)
if __name__=='__main__':main()
