import os
import subprocess
import sys
from confirmation_experiment.common import ROOT,environment,read,write
from confirmation_experiment.commands import eval_command
from confirmation_experiment.recovery.common import AUDIT

def run():
    assert not (AUDIT/'EVAL_BOUNDARY_PARITY.json').exists()
    for mode in ('continuous','reuse_first'):
        cmd=eval_command(45,'mlr')
        cmd[cmd.index('confirmation_experiment.eval_entry')]='confirmation_experiment.recovery.eval_probe'
        for i,x in enumerate(cmd):
            if x.startswith('+experiment_root='):cmd[i]='+experiment_root='+str(AUDIT/('probe_'+mode))
            if x.startswith('+test_log_file='):cmd[i]='+test_log_file='+str(AUDIT/('NO_ROLLOUT_'+mode+'.log'))
        env=dict(environment(),RECOVERY_PROBE_MODE=mode)
        with (AUDIT/('eval_probe_'+mode+'.log')).open('w') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    a,b=[read(AUDIT/('EVAL_PROBE_'+m+'.json')) for m in ('continuous','reuse_first')]
    assert a['records']==b['records'],'Model/RNG/config boundary mismatch'
    write(AUDIT/'EVAL_BOUNDARY_PARITY.json',{'status':'PASS','seed':45,'arm_checkpoints':'existing MLR ep99',
          'all_ten_pre_call_boundaries_equal':True,'remaining_nine_equal':True,
          'compared':['full policy state','Python/NumPy/torch CPU/CUDA RNG','non-output config','task order','module modes','backend flags'],
          'first_cell_stub_rng_consumption_intentionally_different':True,'robot_trials':0,
          'scope':'Tests native preparation and RNG reset, not an additional robot performance measurement.'})

if __name__=='__main__':run()
