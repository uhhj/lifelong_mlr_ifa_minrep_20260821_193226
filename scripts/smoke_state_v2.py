import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from easydict import EasyDict
from omegaconf import OmegaConf
import verify_goal_seed42 as legacy
from libero.lifelong.modal_contract import validate_flags, validate_contract
from libero.lifelong.runbook_core import atomic_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    configs={phase: EasyDict(OmegaConf.to_container(OmegaConf.load(args.run_dir/('resolved_'+phase+'.yaml')),
                                                   resolve=True)) for phase in ['export','adaptation','eval']}
    for cfg in configs.values():
        validate_flags(cfg)
    cfg=configs['export']
    cfg.experiment_dir=str(args.run_dir/'smoke')
    cfg.feature_dir=str(args.run_dir/'smoke/features')
    cfg.buffer_dir=str(args.run_dir/'smoke/buffer')
    # Real optimizer smoke uses the registered batch metadata, with two bounded windows.
    cfg.train.batch_size=10
    args.pretrained=cfg.pretrain
    legacy.smoke(args,cfg)
    from functools import partial
    from libero.lifelong.datasets import get_dataset, initialize_obs_worker
    from libero.libero.benchmark import get_benchmark
    from torch.utils.data import DataLoader, Subset
    benchmark=get_benchmark('LIBERO_GOAL')(0)
    dataset,_=get_dataset(str(Path(cfg.folder)/benchmark.get_task_demonstration(6)),
                          cfg.data.obs.modality,seq_len=8,demo_limit=10)
    loader=DataLoader(Subset(dataset,[0,1]),batch_size=1,num_workers=12,
                      worker_init_fn=partial(initialize_obs_worker,obs_modality=cfg.data.obs.modality))
    count=0
    for data in loader:
        assert data['actions'].shape==(1,8,7)
        assert torch.isfinite(data['obs']['agentview_rgb']).all()
        count+=1
    assert count==2
    atomic_json(args.run_dir/'spawn_loader_test.json',{'status':'PASS','num_workers':12,
                'real_windows':count,'handle_reopened_in_workers':True})
    contract=json.loads((args.run_dir/'modal_contract.json').read_text())
    validate_contract(contract,contract)
    for bad in [None, {'use_state':False},dict(contract,use_state=False)]:
        try:
            validate_contract(bad,contract)
        except ValueError:
            continue
        raise AssertionError('Old/unverifiable modal cache accepted')
    atomic_json(args.run_dir/'test_gate.json',{'status':'PASS','modal_contract':contract,
        'modal_fingerprint':contract['fingerprint'],'three_configs_boolean_true':True,
        'raw_hdf5_spawn_loader':True,'old_cache_rejection':True,'formal_updates':0,'formal_episodes':0})

if __name__=='__main__':
    main()
