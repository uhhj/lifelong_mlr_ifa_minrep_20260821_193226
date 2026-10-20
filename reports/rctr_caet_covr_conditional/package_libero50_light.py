"""Allowlisted metadata only. Never package model/data/RNG tensors or raw actions."""
import json
import shutil
from pathlib import Path
from l50_exp.common import O,verify,sha,write

def main():
    verify()
    assert (O/"pipeline.exit").read_text().strip()=="0"
    assert (O/"supervisor.exit").read_text().strip()=="0"
    assert json.loads((O/"PROCESS_EXIT_ACCEPTANCE.json").read_text())["status"]=="PASS"
    target=O/"light_evidence"
    assert not target.exists()
    candidates=[]
    for p in O.rglob("*"):
        if not p.is_file() or p.suffix not in (".json",".jsonl",".exit"):continue
        if "light_evidence" in p.parts:continue
        if p.stat().st_size>8*1024**2:raise RuntimeError("Oversize metadata needs explicit review: "+str(p))
        # The suite generates only metadata in these formats; still fail on raw
        # tensor payload keys instead of assuming an extension makes it safe.
        if p.suffix in (".json",".jsonl"):
            values=[json.loads(line) for line in p.read_text().splitlines() if line] if p.suffix==".jsonl" else [json.loads(p.read_text())]
            def inspect(x):
                if isinstance(x,dict):
                    assert not (set(x)&{"password","private_key","actions","last_action","raw_actions",
                        "latent_tensor","state_vector","noise_table","rng_state"}),str(p)
                    for v in x.values():inspect(v)
                elif isinstance(x,list):
                    for v in x:inspect(v)
            for x in values:inspect(x)
        candidates.append(p)
    assert sum(p.stat().st_size for p in candidates)<256*1024**2
    target.mkdir()
    manifest={}
    for p in sorted(candidates):
        rel=p.relative_to(O);dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dst);assert sha(dst)==sha(p)
        manifest[str(rel)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    write(target/"PACKAGE_MANIFEST.json",dict(status="REVIEWED_METADATA_ONLY",files=manifest,
        excludes=["weights","demonstrations","latent","buffer_payload","raw_actions","RNG_binary"]))
if __name__=="__main__":main()
