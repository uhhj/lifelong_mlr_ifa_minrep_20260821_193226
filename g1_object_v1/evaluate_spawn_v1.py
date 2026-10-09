"""Evaluation-only bootstrap repair: also bind namespaces in spawned workers.

The frozen evaluator calls bind() inside main(), which multiprocessing spawn
does not execute when importing __mp_main__ before unpickling LIBERO workers.
No policy, action, initial-state, rollout, or metric behavior is changed.
"""
from .bindings import bind
bind()


def _smoke_worker(conn, wrapper_type):
    import libero.libero.envs.venv as envs
    import libero.lifelong.metric as metric
    conn.send(dict(envs=envs.__file__,metric=metric.__file__,
                   wrapper_identity=wrapper_type is envs.CloudpickleWrapper))
    conn.close()


def smoke():
    import multiprocessing as mp
    from .io import write
    from .prepare import OUT
    from libero.libero.envs.venv import CloudpickleWrapper
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe()
    p=ctx.Process(target=_smoke_worker,args=(child,CloudpickleWrapper));p.start();child.close()
    assert parent.poll(45),'spawn import smoke timed out'
    result=parent.recv();p.join(10)
    assert p.exitcode==0 and result['wrapper_identity']
    write(OUT/'EVAL_SPAWN_IMPORT_CHECK.json',dict(status='PASS',child_exitcode=p.exitcode,**result))


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--spawn-import-smoke']:
        smoke()
    else:
        from .evaluate import main
        main()
