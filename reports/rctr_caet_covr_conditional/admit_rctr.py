"""Last read-only launch gate and explicit external resource-binding receipt."""
import hashlib
import json
import subprocess
import time
from pathlib import Path
from coreset_exp.common import O, S, verify, write, sha

verify()
W = S.parent
assert not (O/"pipeline.exit").exists()
assert not (O/"pipeline.pid.json").exists()
a = json.loads((O/"GPU_PREFLIGHT.json").read_text())
assert a["status"] == "PASS" and a["optimizer_updates"] == 0
assert a["engineering_steps"] == 28 and a["formal_episodes"] == 0
assert (W/"preflight.exit").read_text().strip() == "1"
assert (W/"preflight2.exit").read_text().strip() == "0"
assert "FileNotFoundError" in (W/"preflight.log").read_text()
assert (S/"libero/libero/assets").resolve() == Path("/data/mlr_ifa/src/libero/libero/assets")
gpu = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"], text=True).strip()
assert not gpu, "GPU already occupied: "+gpu
write(O/"RESOURCE_BINDING.json", dict(status="PASS",
    asset_path=str((S/"libero/libero/assets").resolve()),
    asset_binding="Existing unchanged shared native assets, same as RA parent",
    first_preflight_exit=1, first_preflight_log_sha256=sha(W/"preflight.log"),
    failure="Missing non-code asset symlink, before first env step and before any optimizer update",
    repaired_preflight_exit=0, repaired_preflight_log_sha256=sha(W/"preflight2.log"),
    frozen_source_unchanged=True, unix=time.time()))
write(O/"LAUNCH_ACCEPTANCE.json", dict(status="PASS", unix=time.time(),
    original_contract_sha256=sha(O/"CONTRACT.json"),
    preflight_sha256=sha(O/"GPU_PREFLIGHT.json"),
    source_count=len(json.loads((O/"CONTRACT.json").read_text())["source_sha256"]),
    orchestration={str(p):sha(p) for p in (W/"env.sh", W/"supervise.sh")},
    no_existing_gpu_worker=True, new_training_started=False,
    real_sidecar_backprop="Deferred until certified records exist; required before Task7+ uses them"))
print("LAUNCH_ACCEPTANCE PASS")
