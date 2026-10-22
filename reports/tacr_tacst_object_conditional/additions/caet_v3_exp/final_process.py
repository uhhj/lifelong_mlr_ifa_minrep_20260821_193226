"""External supervisor calls this only after pipeline exited and was reaped."""
import json
import os
import time
from pathlib import Path
from coreset_exp.common import O, S, verify, write, sha


def main():
    verify()
    assert (O/"pipeline.exit").read_text().strip() == "0"
    assert json.loads((O/"AUDIT.json").read_text())["status"] == "PASS"
    residual = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(p.name)
            if pid in (os.getpid(), os.getppid()):
                continue
            cmd = (p/"cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            cwd = (p/"cwd").resolve()
            if cwd == S and any(x in cmd for x in (
                    "caet_v3_exp.train", "caet_v3_exp.collect", "coreset_exp.evaluate", "caet_v3_exp.evaluate",
                    "caet_v3_exp.pipeline", "torch.distributed.run")):
                residual.append(dict(pid=pid, command=cmd))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    assert not residual, residual
    write(O/"PROCESS_EXIT_ACCEPTANCE.json", dict(status="PASS", unix=time.time(),
        pipeline_exit=0, residual_workers=[], audit_sha256=sha(O/"AUDIT.json")))
    write(O/"RUN_STATUS.json", dict(status="COMPLETED", unix=time.time(),
        metric_gate=json.loads((O/"AUDIT.json").read_text())["metric_gate"]))


if __name__ == "__main__":
    main()
