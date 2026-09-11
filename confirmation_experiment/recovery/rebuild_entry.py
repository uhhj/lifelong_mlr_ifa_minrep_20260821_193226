"""Reconstruct a lost live boundary, verify it, then continue the same process."""
import os
import runpy
from pathlib import Path
from confirmation_experiment.common import (ROOT,RUNS,REPORTS,read,write,sha,rng_state,state_digest,
                                           state_to_cpu,atomic_torch_save,exact)
from confirmation_experiment.recovery.common import WORK,AUDIT,install_abort_routing

def main():
    install_abort_routing()
    import confirmation_experiment.instrumentation as instrument
    from confirmation_experiment.common import wait_for
    original=instrument.ConfirmationSoft.learn_all_tasks
    original_start=instrument.ConfirmationSoft.start_task
    target=RUNS/'formal/seed45/soft'
    expected=read(target/'TASK6_TRAIN_READY.json')
    class_state={'recovered':False}
    def learned(self,datasets,benchmark,summary,task):
        if task!=6:
            assert class_state['recovered'];return original(self,datasets,benchmark,summary,task)
        self.integrity_barriers=False
        result=original(self,datasets,benchmark,summary,task)
        rebuilt=read(self.confirmation_root/'TASK6_TRAIN_READY.json')
        exact(expected,rebuilt,'original vs reconstructed task6')
        # Dataset and replay copies must retain their original names and bytes.
        rebuilt_buffer=Path(self.cfg.buffer_dir);original_cfg=read(target/'run_001/config.json')
        original_buffer=Path(original_cfg['buffer_dir'])
        a={p.name:sha(p) for p in rebuilt_buffer.glob('*.pt')}
        b={p.name:sha(p) for p in original_buffer.glob('*.pt')}
        assert a==b and len(a)>0
        old_events=target/'events.jsonl';count=0
        with old_events.open() as f:
            import json
            for line in f:
                e=json.loads(line);count+=1
                assert e['current_task']==6 and e['global_optimizer_step']==count
        assert count==self._confirm_global==expected['batches']==15700
        state={'model':state_to_cpu(self.policy.state_dict()),'optimizer':state_to_cpu(self.optimizer.state_dict()),
               'scheduler':state_to_cpu(self.scheduler.state_dict()),'rng':rng_state(),
               'optimizer_step':self._optimizer_step_count,'scheduler_step':self._scheduler_step_count,
               'global_optimizer_step':self._confirm_global,'buffer':b,'original_config':original_cfg,
               'original_witness':expected,'training_modes':{n:m.training for n,m in self.policy.named_modules()},
               'boundary':'Original learn_all_tasks(task6) returned after buffer update; before task7 construction.'}
        atomic_torch_save(WORK/'soft45_boundary.pt',state)
        if self._confirm_events:self._confirm_events.flush();self._confirm_events.close()
        for k in ('experiment_root','experiment_dir','experiment_name','buffer_dir'):setattr(self.cfg,k,original_cfg[k])
        exact(dict(self.cfg),original_cfg,'resumed configuration')
        self.experiment_dir=original_cfg['experiment_dir'];self.confirmation_root=target
        self._confirm_events=old_events.open('a',encoding='utf-8',buffering=1024*1024)
        self._confirm_timings=read(target/'instrumentation_timing.json')
        assert state_digest(rng_state())==expected['rng_sha256']
        class_state['recovered']=True
        write(AUDIT/'SOFT45_BOUNDARY.json',{'status':'PASS','all_original_witness_fields_equal':True,
              'matched':rebuilt,'replay_files':len(b),'original_task6_events_retained':count,
              'state_sha256':sha(WORK/'soft45_boundary.pt'),'continuation':'Same live process after exact prefix reconstruction',
              'reconstruction_only_directory':str(WORK/'rebuild_soft45')})
        wait_for(WORK/'RELEASE_SOFT45',WORK/'FAILED.json')
        assert state_digest(rng_state())==expected['rng_sha256'],'Waiting changed RNG'
        return result
    def start_task(self,task,loader):
        result=original_start(self,task,loader)
        if class_state['recovered'] and self._confirm_stage==7:
            reference=read(RUNS/'formal/seed45/mlr/plans/task7.json')
            for k in ('record_sha256','immutable_feature_origins','sampler','batch_size','drop_last','epochs'):
                exact(self._confirm_plan[k],reference[k],'task7 original sampling '+k)
            write(AUDIT/'SOFT45_TASK7_SAMPLING.json',{'status':'PASS','same_ordered_records_and_sampler_as_original_mlr':True})
        return result
    instrument.ConfirmationSoft.learn_all_tasks=learned
    instrument.ConfirmationSoft.start_task=start_task
    runpy.run_module('confirmation_experiment.train_entry',run_name='__main__')

if __name__=='__main__':main()
