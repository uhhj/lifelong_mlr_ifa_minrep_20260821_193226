"""Repair only the independent test expectation; original native head/rollout remain frozen."""
import inspect,hashlib,json,time
from pathlib import Path
import screen
W=screen.W;O=screen.O
assert (W/"preflight.exit").read_text().strip()=="1"
assert not (W/"pipeline.started").exists()
assert not (O/"parent9").exists() and not (O/"teacher7").exists()
screen.verify()
code=inspect.getsource(screen.preflight)
old='means=head.mean_layer(x).view(-1,5,7)'
new='means=torch.tanh(head.mean_layer(x).view(-1,5,7))'
assert code.count(old)==1
record=dict(time=time.time(),reason="Independent test omitted native GMMHead.forward_fn tanh on means.",
    original_preflight_exit=1,original_screen_sha256=screen.sha(W/"screen.py"),
    original_contract_sha256=screen.sha(O/"CONTRACT.json"),
    adapter_sha256=screen.sha(__file__),old_expression=old,new_expression=new,
    original_scientific_code_changed=False,rollout_changed=False,optimizer_calls=0,environment_calls_before_repair=0)
screen.write(O/"PREFLIGHT_REPAIR.json",record)
exec(compile(code.replace(old,new),str(__file__),"exec"),screen.__dict__)
screen.preflight()
screen.verify()
