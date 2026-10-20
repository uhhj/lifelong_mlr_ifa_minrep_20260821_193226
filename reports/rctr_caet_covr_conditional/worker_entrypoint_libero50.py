"""Serial LIBERO50 worker: shared disclosed pretraining once, then one seed."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from worker_entrypoint import install
from queue_controller import write
from queue_verifier import sha

def run(e,w,out,pretraining=False):
    src=w/"source"
    assert not out.exists() and not (w/"admission").exists()
    w.mkdir(parents=True,exist_ok=True)
    install(e["bundle"],src,e["bundle_sha256"])
    q=Path(e["queue"])
    for name in ("PARTITION","PARENT_EXPORT_TEMPLATE","PARENT_TRAIN_TEMPLATE","PARENT_EVAL_TEMPLATE"):
        source=q/("LIBERO50_"+name+".json")
        assert sha(source)==e["resource_hashes"][str(source)]
        (w/(name+".json")).write_bytes(source.read_bytes())
    env=dict(os.environ,LIBERO50_WORKSPACE=str(w),LIBERO50_OUT=str(out),
        LIBERO50_SEED="42" if pretraining else str(e["seed"]),
        LIBERO50_PRETRAIN_OUT=e["shared_pretrain_output"],PYTHONPATH=str(src),
        HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="0",MUJOCO_GL="egl",EGL_PLATFORM="surfaceless",
        OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
    admission=w/"admission";admission.mkdir()
    def phase(name,argv):
        assert not (admission/(name+".exit")).exists()
        write(admission/(name+".command.json"),dict(argv=argv,unix=time.time()))
        with (admission/(name+".log")).open("x") as f:
            p=subprocess.Popen(argv,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            write(admission/(name+".pid.json"),dict(pid=p.pid,unix=time.time()))
            rc=p.wait()
        (admission/(name+".exit")).write_text(str(rc)+"\n")
        assert rc==0,(name,rc)
    # CPU tensors only; valid EGL mapping is retained for native import side effects.
    phase("software",[sys.executable,"test_libero50.py"])
    phase("ra_software",[sys.executable,"-m","unittest","ra_exp.test_reference"])
    phase("prepare",[sys.executable,"-m","l50_exp.prepare"]+(["--pretrain"] if pretraining else []))
    probe="pretrain_preflight" if pretraining else "preflight"
    phase("gpu_preflight",[sys.executable,"-m","torch.distributed.run","--standalone",
        "--nproc_per_node=1","-m","l50_exp."+probe])
    report="PRETRAIN_GPU_PREFLIGHT.json" if pretraining else "GPU_PREFLIGHT.json"
    assert json.loads((out/report).read_text())["status"]=="PASS"
    write(out/"LAUNCH_ACCEPTANCE.json",dict(status="PASS",unix=time.time(),
        bundle_sha256=e["bundle_sha256"],contract_sha256=sha(out/"CONTRACT.json"),
        gpu_preflight_sha256=sha(out/report),software_exits=[0,0],prepare_exit=0))
    (out/"supervisor.pid").write_text(str(os.getpid())+"\n")
    rc=1
    try:
        phase("pipeline",[sys.executable,"-u","-m","l50_exp.pipeline"])
        phase("final_process",[sys.executable,"-m","l50_exp.final_process"])
        rc=0
    finally:(out/"supervisor.exit").write_text(str(rc)+"\n")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("entry");a=ap.parse_args()
    e=json.loads(Path(a.entry).read_text())
    assert e["method"]=="RA_IFA" and e["suite"]=="LIBERO50" and e["seed"] in (41,42,43)
    q=Path(e["queue"])
    status=json.loads((q/"LIBERO50_SERVER_STATUS.json").read_text())
    assert status["status"]=="DATA_VERIFIED" and status["verified"]==50
    shared=Path(e["shared_pretrain_output"])
    if not shared.exists():
        run(e,Path(e["shared_pretrain_workspace"]),shared,True)
    else:
        # Existing incomplete pretraining is never rerun or silently discarded.
        assert (shared/"supervisor.exit").read_text().strip()=="0"
        assert (shared/"pipeline.exit").read_text().strip()=="0"
        accepted=json.loads((shared/"PROCESS_EXIT_ACCEPTANCE.json").read_text())
        strict=json.loads((shared/"PRETRAIN_STRICT_LOAD.json").read_text())
        assert accepted["status"]==strict["status"]=="PASS"
        assert accepted["audit_sha256"]==sha(shared/"PRETRAIN_STRICT_LOAD.json")
        assert sha(strict["checkpoint"])==strict["sha256"]
        contract=json.loads((shared/"CONTRACT.json").read_text())
        for p,h in contract["files"].items():assert sha(p)==h,p
    run(e,Path(e["source"]).parent,Path(e["output"]))
if __name__=="__main__":main()
