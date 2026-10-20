"""One admitted run, no algorithmic retries. Serial controller owns the GPU."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from queue_verifier import sha
from queue_controller import write


def install(bundle_path, target, expected):
    assert sha(bundle_path) == expected
    assert not target.exists()
    target.mkdir(parents=True)
    bundle=json.loads(Path(bundle_path).read_text())
    for rel,item in bundle.items():
        path=(target/rel).resolve()
        assert target.resolve() in path.parents
        assert path.suffix in (".py",".yaml",".yml",".sh",".json",".md")
        raw=item["text"].encode()
        assert hashlib.sha256(raw).hexdigest()==item["sha256"]
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    assets=target/"libero/libero/assets"
    assert not assets.exists()
    assets.symlink_to("/data/mlr_ifa/src/libero/libero/assets",target_is_directory=True)
    assert assets.is_dir()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("entry");args=ap.parse_args()
    e=json.loads(Path(args.entry).read_text())
    method=e["method"].lower()
    assert method in ("rctr","caet","covr") and e["suite"]=="GOAL"
    w=Path(e["source"]).parent;out=Path(e["output"]);src=Path(e["source"])
    assert not out.exists() and not (w/"admission").exists()
    w.mkdir(parents=True,exist_ok=True)
    install(e["bundle"],src,e["bundle_sha256"])
    env=dict(os.environ,CORESET_SOURCE=str(src),CORESET_OUT=str(out),
             CORESET_SEED=str(e["seed"]),RA_BASE=e["baseline"],PYTHONPATH=str(src),
             HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONDONTWRITEBYTECODE="1",
             CUDA_VISIBLE_DEVICES="0",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",
             OMP_NUM_THREADS="4",MUJOCO_GL="egl",EGL_PLATFORM="surfaceless")
    admission=w/"admission";admission.mkdir()
    def phase(name,argv,cpu=False):
        assert not (admission/(name+".exit")).exists()
        write(admission/(name+".command.json"),dict(argv=argv,unix=time.time()))
        with (admission/(name+".log")).open("x") as log:
            proc=subprocess.Popen(argv,cwd=src,env=dict(env,CUDA_VISIBLE_DEVICES="") if cpu else env,
                                  stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            write(admission/(name+".pid.json"),dict(pid=proc.pid,unix=time.time()))
            rc=proc.wait()
        (admission/(name+".exit")).write_text(str(rc)+"\n")
        assert rc==0,(name,rc)
    try:
        phase("software",[sys.executable,str(src/"test_auxiliary.py")],cpu=True)
        phase("prepare",[sys.executable,"-m",method+"_exp.prepare"],cpu=True)
        if method=="caet":
            phase("labels6",[sys.executable,"-m","caet_exp.labels","6"],cpu=True)
        phase("gpu_preflight",[sys.executable,"-m","torch.distributed.run","--standalone",
             "--nproc_per_node=1","-m",method+"_exp.preflight"])
        assert json.loads((out/"GPU_PREFLIGHT.json").read_text())["status"]=="PASS"
        write(out/"LAUNCH_ACCEPTANCE.json",dict(status="PASS",unix=time.time(),
            contract_sha256=sha(out/"CONTRACT.json"),
            source_bundle_sha256=e["bundle_sha256"],
            entry_sha256=sha(args.entry),
            gpu_preflight_sha256=sha(out/"GPU_PREFLIGHT.json"),
            software_exit=0,prepare_exit=0,preflight_exit=0))
        (out/"supervisor.pid").write_text(str(os.getpid())+"\n")
        rc=1
        try:
            phase("pipeline",[sys.executable,"-u","-m",method+"_exp.pipeline"])
            phase("final_process",[sys.executable,"-m",method+"_exp.final_process"],cpu=True)
            rc=0
        finally:
            (out/"supervisor.exit").write_text(str(rc)+"\n")
    except Exception as exc:
        write(w/"ENTRY_FAILURE.json",dict(error=repr(exc),traceback=traceback.format_exc(),
                                          unix=time.time(),all_original_logs_preserved=True))
        raise


if __name__=="__main__":
    main()
