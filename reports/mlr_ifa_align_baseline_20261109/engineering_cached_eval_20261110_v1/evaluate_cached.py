"""Original evaluator with only accepted-cache get_action override."""
from runtime_cached import *
import fcntl,runpy
def main():
    case=sys.argv[1];assert case in CASES;verify()
    lock=(Q/'environment.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    event('environment_acquired',case=case,pid=os.getpid());code=1
    try:
        sys.path.insert(0,str(Q/'cases'/case/'source'))
        import torch
        from coreset_exp import model,common
        from libero.lifelong.models.complete_policy import CompletePolicy
        from cache_policy import cached_get_action,admission
        context={};original_setup=model.data_setup
        def checked_setup(*args,**kwargs):
            result=original_setup(*args,**kwargs);context['benchmark']=result[0];return result
        model.data_setup=checked_setup
        original_load=model.strict_downstream
        def checked_load(p,path):
            result=original_load(p,path)
            assert isinstance(p,CompletePolicy)
            saved=common.rng()
            try:
                instr,mask=context['benchmark'].get_task_emb(8)
                proof=admission(p,p.shape_meta,instr,mask,common.rng,common.restore_rng)
            finally:common.restore_rng(saved)
            from runtime_cached import sha,write
            proof.update(checkpoint_sha256=sha(path),case=case,observed=time.time(),source='S evaluator cache admission; original global RNG restored')
            # One proof per actual cell load, independent of model stage reuse.
            out=E/'admissions'/case;out.mkdir(parents=True,exist_ok=True)
            idx=len(list(out.glob('*.json')))
            write(out/('%02d.json'%idx),proof)
            return result
        model.strict_downstream=checked_load
        CompletePolicy.get_action=cached_get_action
        sys.argv=['coreset_exp.evaluate','random']
        runpy.run_module('coreset_exp.evaluate',run_name='__main__',alter_sys=True);code=0
    finally:
        event('environment_released',case=case,pid=os.getpid(),code=code)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
