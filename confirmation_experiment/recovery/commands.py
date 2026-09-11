from confirmation_experiment.commands import train_command,eval_command
from confirmation_experiment.recovery.common import WORK

def isolated(command,run_id):
    c=list(command);i=c.index('--standalone')
    c[i:i+1]=['--rdzv_backend=c10d','--rdzv_endpoint=localhost:0','--rdzv_id='+run_id]
    return c

def train(seed,arm):
    c=isolated(train_command(seed,arm),'confirmation-recovery-seed%d-%s'%(seed,arm))
    c[c.index('confirmation_experiment.train_entry')]='confirmation_experiment.recovery.train_entry'
    return c

def rebuild_soft45():
    c=train(45,'soft');c[c.index('confirmation_experiment.recovery.train_entry')]='confirmation_experiment.recovery.rebuild_entry'
    for i,s in enumerate(c):
        if s.startswith('+experiment_root='):c[i]='+experiment_root='+str(WORK/'rebuild_soft45')
        if s.startswith('+buffer_dir='):c[i]='+buffer_dir='+str(WORK/'rebuild_soft45_buffer')
    return c

def evaluate(seed,arm,reuse_first=False):
    c=eval_command(seed,arm)
    c[c.index('confirmation_experiment.eval_entry')]='confirmation_experiment.recovery.eval_entry'
    if reuse_first:
        for i,s in enumerate(c):
            if s.startswith('+experiment_root='):c[i]='+experiment_root='+str(WORK/f'eval_seed{seed}_{arm}')
            if s.startswith('+test_log_file='):c[i]='+test_log_file='+str(WORK/f'primary_seed{seed}_{arm}_segment.log')
    return c
