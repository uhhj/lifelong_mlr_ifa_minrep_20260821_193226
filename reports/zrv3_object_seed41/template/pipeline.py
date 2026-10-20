"""Finite background pipeline; no baseline execution or automatic blind restart."""
import os,sys,time,json,subprocess,traceback
from pathlib import Path
W=Path(__file__).resolve().parent;O=Path(json.loads((W/'RUN_SPEC.json').read_text())['output'])
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    assert not (O/'pipeline.started').exists() and not (O/'pipeline.exit').exists()
    write(O/'pipeline.started',dict(pid=os.getpid(),time=time.time()));(O/'pipeline.pid').write_text(str(os.getpid()))
    code=1
    try:
        for phase in ['calibrate','evaluate','audit']:
            assert not (O/(phase+'.exit')).exists()
            command=[sys.executable,'-u','-m','zr3_exp.'+phase];start=time.time()
            with (O/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,command=command,time=start))+'\n')
            with (O/(phase+'.log')).open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            (O/(phase+'.exit')).write_text(str(r.returncode)+'\n')
            with (O/'timings.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,seconds=time.time()-start,exit=r.returncode))+'\n')
            if r.returncode:raise RuntimeError((phase,r.returncode))
        code=0
    except BaseException:write(O/'PIPELINE_FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
    finally:(O/'pipeline.exit').write_text(str(code)+'\n')
    return code
if __name__=='__main__':raise SystemExit(main())
