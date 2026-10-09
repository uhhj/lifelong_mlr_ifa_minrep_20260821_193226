"""Import-only spawn check. No environment construction or rollout."""
import json
import multiprocessing as mp

def child(q):
    import libero.libero.envs.venv as v
    q.put(dict(module=v.__file__,worker_imported=hasattr(v,'SubprocVectorEnv')))

if __name__=='__main__':
    ctx=mp.get_context('spawn');q=ctx.Queue();p=ctx.Process(target=child,args=(q,))
    p.start();result=q.get(timeout=45);p.join(timeout=10)
    assert p.exitcode==0 and result['worker_imported']
    print(json.dumps(dict(status='PASS',child_exit=p.exitcode,environment_instances=0,rollouts=0,**result)))
