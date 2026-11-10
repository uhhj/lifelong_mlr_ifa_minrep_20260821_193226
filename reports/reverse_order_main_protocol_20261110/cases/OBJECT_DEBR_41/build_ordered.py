"""Exact native factory globals; only an explicit order argument is changed."""
import inspect,hashlib,ast
from caet_v3_exp import train as _native
assert hashlib.sha256(open(_native.__file__,'rb').read()).hexdigest()=='8e6ae52166f00c06361e21d9ae422dc95d922ca2be41f64c11fc73ffcc30706f'
_source=inspect.getsource(_native.build).strip()
assert _source=="def build(c):\n    control_seed(SEED);b=get_benchmark('LIBERO_OBJECT')(0)\n    b.set_task_embs(get_task_embs(c,[b.get_task(i).language for i in range(10)]))\n    Path(c.experiment_dir).mkdir(parents=True,exist_ok=True);Path(c.buffer_dir).mkdir(parents=True,exist_ok=True)\n    a=V3MLR(1,c).cuda();n=strict_downstream(a.policy,c.pretrain)\n    assert all(n.startswith(('temporal_transformer.','policy_head.')) for n,p in a.policy.named_parameters() if p.requires_grad)\n    a.policy=DDP(a.policy,device_ids=[0],output_device=0);a.train();a.step=0\n    return a,b"
_ordered="def build(c):\n    control_seed(SEED);b=get_benchmark('LIBERO_OBJECT')(c.data.task_order_index)\n    b.set_task_embs(get_task_embs(c,[b.get_task(i).language for i in range(10)]))\n    Path(c.experiment_dir).mkdir(parents=True,exist_ok=True);Path(c.buffer_dir).mkdir(parents=True,exist_ok=True)\n    a=V3MLR(1,c).cuda();n=strict_downstream(a.policy,c.pretrain)\n    assert all(n.startswith(('temporal_transformer.','policy_head.')) for n,p in a.policy.named_parameters() if p.requires_grad)\n    a.policy=DDP(a.policy,device_ids=[0],output_device=0);a.train();a.step=0\n    return a,b"
_globals=dict(vars(_native));exec(compile(_ordered,__file__,'exec'),_globals)
build=_globals['build']
FACTORY_PROOF=dict(parent_sha256='8e6ae52166f00c06361e21d9ae422dc95d922ca2be41f64c11fc73ffcc30706f',parent_source=_source,ordered_source=_ordered,order_argument_from_configuration=True,model_loss_optimizer_unchanged=True,source_shared_file_modified=False)
