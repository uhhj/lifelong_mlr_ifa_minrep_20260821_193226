"""Single authorized seed, finite sequential train/eval/audit; no automatic retry."""
from pathlib import Path
import os,sys,json,time,subprocess,traceback
W=Path(__file__).resolve().parent;O=Path(os.environ['CORESET_OUT'])
def main():
    assert not (O/'pipeline.started').exists();(O/'pipeline.started').write_text(json.dumps(dict(pid=os.getpid(),time=time.time())))
    (O/'pipeline.pid').write_text(str(os.getpid()));code=1
    try:
        jobs=[('train',[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m','coreset_exp.train','sc_coreset']),('evaluate',[sys.executable,'-u','-m','coreset_exp.evaluate','sc_coreset']),('audit',[sys.executable,'-u','-m','sc_exp.audit'])]
        for name,cmd in jobs:
            assert not (O/(name+'.exit')).exists();start=time.time()
            with (O/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(phase=name,command=cmd,time=start))+'\n')
            with (O/(name+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            (O/(name+'.exit')).write_text(str(r.returncode)+'\n')
            with (O/'timings.jsonl').open('a') as f:f.write(json.dumps(dict(phase=name,seconds=time.time()-start,exit=r.returncode))+'\n')
            if r.returncode:raise RuntimeError((name,r.returncode))
        code=0
    except BaseException:(O/'PIPELINE_FAILURE.json').write_text(json.dumps(dict(time=time.time(),traceback=traceback.format_exc())))
    finally:(O/'pipeline.exit').write_text(str(code)+'\n')
    return code
if __name__=='__main__':raise SystemExit(main())
