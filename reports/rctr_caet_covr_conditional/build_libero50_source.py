"""New suite source, explicit constructor adaptation; no mutation of active workers."""
from pathlib import Path
import hashlib
import json

R=Path(__file__).resolve().parent
def main():
    bundle={}
    for p in (R/"source").rglob("*"):
        if p.is_file() and p.suffix in (".py",".yaml",".yml",".sh") and "__pycache__" not in p.parts:
            bundle[p.relative_to(R/"source").as_posix()]=p.read_text(encoding="utf8")
    key="libero/lifelong/models/complete_policy.py";old=bundle[key]
    needle="        if self.init == 1:\n            self.init = 0\n            self.load_state_dict(torch_load_model(cfg.pretrain)[0])"
    assert old.count(needle)==1
    new=old.replace(needle,'''        if getattr(cfg, "pretraining_mode", False):
            assert cfg.pretrain is None, "New pretraining must not consume task checkpoint"
            for name, param in self.named_parameters():
                param.requires_grad = (".lora_" in name or name.startswith((
                    "image_proj.", "film_scale.", "film_shift.", "state_encoder.",
                    "state_scale.", "state_shift.", "temporal_transformer.", "policy_head.")))
            self.init = 0
        elif self.init == 1:
            self.init = 0
            self.load_state_dict(torch_load_model(cfg.pretrain)[0])''')
    bundle[key]=new
    for p in (R/"libero50_additions").rglob("*.py"):
        if "__pycache__" not in p.parts:bundle[p.relative_to(R/"libero50_additions").as_posix()]=p.read_text(encoding="utf8")
    original_rollout=(R/"source/coreset_exp/eval_metric.py").read_text()
    adapted_rollout=original_rollout.replace("from coreset_exp.evaluate import ledger",
                                             "from l50_exp.evaluate import ledger")
    assert adapted_rollout!=original_rollout
    bundle["l50_exp/eval_metric.py"]=adapted_rollout
    proof=dict(status="DRAFT_NOT_GPU_ADMITTED",base_constructor_sha256=hashlib.sha256(old.encode()).hexdigest(),
        adapted_constructor_sha256=hashlib.sha256(new.encode()).hexdigest(),
        delta="explicit pretraining flag enables LoRA/new projection/FiLM/state/temporal/head; no checkpoint loaded",
        downstream_false_flag="same constructor and strict task checkpoint loading",
        no_in_place_change=True)
    out=R/"libero50_preparation/source"/key;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(new,encoding="utf8",newline="\n")
    packed={n:dict(text=t,sha256=hashlib.sha256(t.encode()).hexdigest()) for n,t in bundle.items()}
    raw=(json.dumps(packed)+"\n").encode()
    (R/"LIBERO50_SOURCE_BUNDLE.json").write_bytes(raw)
    proof["bundle_sha256"]=hashlib.sha256(raw).hexdigest();proof["files"]=len(packed)
    (R/"libero50_preparation/SOURCE_ADAPTATION.json").write_text(json.dumps(proof,indent=2)+"\n")
    print(json.dumps(proof))
if __name__=="__main__":main()
