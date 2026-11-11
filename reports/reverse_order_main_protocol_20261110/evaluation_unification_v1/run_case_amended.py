"""Enter one inherited method; no automatic resume or outcome retry."""
from pathlib import Path
import argparse,contextlib,json,os,sys
Q=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('case');ap.add_argument('phase',choices=['train','evaluate']);args=ap.parse_args()
R=Q/'cases'/args.case
inp=json.loads((R/'INPUT.json').read_text());os.environ.update(REVERSE_CASE_ROOT=str(R),CORESET_OUT=inp['input_root'],CORESET_SEED=str(inp['seed']))
sys.path.insert(0,str(Q));sys.path.insert(0,str(R))
if inp['method']=='MLR':sys.path.insert(0,str(R/'source'))
else:sys.path.insert(0,inp['source_root']);os.environ['CAET_VARIANT']='ALIGN'
from order import install,verify_mapping
install()
from reverse_common import verify_case,write,sha
verify_case(R)
if inp['method']=='MLR':
    from coreset_exp.model import cfg
    c=cfg('random' if args.phase=='train' else 'eval')
    mapping=verify_mapping(c);write(R/'ACTUAL_ORDER.json',dict(status='PASS_ACTUAL_BENCHMARK_MAPPING',mapping=mapping,base_unchanged=True))
    if args.phase=='train':
        import torch,instrumentation
        instrumentation.install()
        assert torch.cuda.is_available()
        write(Path(inp['input_root'])/'GPU_PREFLIGHT.json',dict(status='PASS',scope='ACTUAL_HARDWARE_ONLY_NOT_STAGE_ADMISSION',gpu=torch.cuda.get_device_name(0),torch=torch.__version__))
        from coreset_exp import train
        sys.argv=['coreset_exp.train','random'];ok=False
        try:train.main();ok=True
        finally:
            write(R/'OPTIMIZER_CALLS.json',dict(status='ACTUAL_CALLS_COMPLETED' if ok else 'ACTUAL_CALLS_FAILED_PARTIAL',calls=instrumentation.CALLS,extra_autograd=instrumentation.AUTOGRAD,environments=0))
        assert instrumentation.CALLS==dict(production=inp['physical_budget'],admission=20)
    else:
        sys.path.insert(0,str(Q/'evaluation_unification_v1'))
        from unified_mlr_entry import main as evaluate_common
        evaluate_common()
else:
    from easydict import EasyDict
    mapping=verify_mapping(EasyDict(inp['train_config'] if args.phase=='train' else inp['eval_config']))
    write(R/'ACTUAL_ORDER.json',dict(status='PASS_ACTUAL_BENCHMARK_MAPPING',mapping=mapping,base_unchanged=True))
    if args.phase=='train':
        import instrumentation
        instrumentation.install()
        import train
        original_admit=train.admit
        def admit(*a,**kw):
            with instrumentation.admission():return original_admit(*a,**kw)
        train.admit=admit
        original_gate=train.gate_admission.admission
        def gate(*a,**kw):
            with instrumentation.admission():return original_gate(*a,**kw)
        train.gate_admission.admission=gate
        sys.argv=['train.py','S'];ok=False
        try:train.main();ok=True
        finally:
            write(R/'OPTIMIZER_CALLS.json',dict(status='ACTUAL_CALLS_COMPLETED' if ok else 'ACTUAL_CALLS_FAILED_PARTIAL',calls=instrumentation.CALLS,extra_autograd=instrumentation.AUTOGRAD,environments=40 if ok else 'SEE_ACTUAL_COLLECTION_ATTEMPTS'))
        assert instrumentation.CALLS==dict(production=inp['physical_budget'],admission=91)
    else:
        import evaluate
        sys.argv=['evaluate.py'];evaluate.main()
verify_case(R)
