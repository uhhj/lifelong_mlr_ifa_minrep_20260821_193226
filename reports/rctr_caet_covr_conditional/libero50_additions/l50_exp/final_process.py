"""Read-only exit audit, called after the pipeline has been reaped."""
import json
import os
import time
from pathlib import Path
from .common import O,S,verify,write,sha

def main():
    c=verify()
    assert (O/"pipeline.exit").read_text().strip()=="0"
    report=O/("PRETRAIN_STRICT_LOAD.json" if c["pretraining"] else "AUDIT.json")
    assert json.loads(report.read_text())["status"]=="PASS"
    residual=[]
    for p in Path("/proc").glob("[0-9]*"):
        try:
            if int(p.name) in (os.getpid(),os.getppid()):continue
            cmd=(p/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
            env=(p/"environ").read_bytes().split(b"\0")
            if ("LIBERO50_OUT="+str(O)).encode() not in env:continue
            if any(x in cmd for x in ("l50_exp.pretrain","l50_exp.train","l50_exp.export",
                    "l50_exp.evaluate","l50_exp.pipeline","torch.distributed.run")):
                residual.append(dict(pid=int(p.name),command=cmd))
        except (OSError,ProcessLookupError):pass
    assert not residual,residual
    write(O/"PROCESS_EXIT_ACCEPTANCE.json",dict(status="PASS",pipeline_exit=0,
        residual_workers=[],audit_sha256=sha(report),contract_sha256=sha(O/"CONTRACT.json"),
        unix=time.time(),pretraining=c["pretraining"]))
    write(O/"RUN_STATUS.json",dict(status="COMPLETED",unix=time.time(),pretraining=c["pretraining"]))
if __name__=="__main__":main()
