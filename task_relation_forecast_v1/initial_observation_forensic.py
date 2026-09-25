"""Capture the first native initialization only; no policy actions or scored trials."""
import argparse
import os
import sys
import numpy as np
from .common import OUT,write,sha
from .behavior import cells,run_cell


class CaptureFinished(Exception):
    pass


def main(attempt):
    import paper_qp_rollout_v1.evaluate as evaluator
    from paper_qp_rollout_v1.common import observation_digest
    Original=evaluator.Observer
    destination=OUT/'behavior_attempts/s45/source8/task7'/('attempt'+attempt)
    class CaptureObserver(Original):
        def set_init_state(self,states):
            obs=super().set_init_state(states)
            arrays={};metadata=[]
            def visit(value,path):
                if isinstance(value,np.ndarray) and value.dtype.hasobject:
                    for i,item in enumerate(value.flat):visit(item,path+'/'+str(i))
                elif isinstance(value,dict):
                    for key in sorted(value):visit(value[key],path+'/'+key)
                elif isinstance(value,(list,tuple)):
                    for i,item in enumerate(value):visit(item,path+'/'+str(i))
                else:
                    array=np.asarray(value);key='a%04d'%len(arrays);arrays[key]=array.copy()
                    metadata.append(dict(path=path,key=key,shape=list(array.shape),dtype=str(array.dtype),
                                         digest=observation_digest(value)))
            visit(obs,'obs')
            tmp=destination/'OBSERVATIONS.tmp.npz'
            np.savez_compressed(tmp,**arrays);os.replace(tmp,destination/'OBSERVATIONS.npz')
            write(destination/'INITIAL_OBSERVATION_CAPTURE.json',dict(initial_obs_sha256=observation_digest(obs),
                  init_sha256=self.active['init_sha256'],policy_rng_before=self.active['policy_rng_before'],
                  python_hash_seed=os.environ.get('PYTHONHASHSEED'),fields=metadata,
                  npz_sha256=sha(destination/'OBSERVATIONS.npz'),native_initial_round=0,
                  policy_actions_executed=0,scored_trials=0,formal_training_resumed=False))
            self.inner.close()
            raise CaptureFinished()
    evaluator.Observer=CaptureObserver
    try:
        run_cell(cells(False)[13],attempt=attempt)
    except CaptureFinished:
        write(destination/'INIT_CAPTURE_COMPLETE.json',dict(status='CAPTURED',formal_training_resumed=False))
    finally:
        evaluator.Observer=Original


if __name__=='__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn',force=True)
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args();main(a.attempt)
