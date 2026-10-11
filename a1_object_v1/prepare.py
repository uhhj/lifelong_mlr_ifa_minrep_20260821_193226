import argparse,json,platform,subprocess
from pathlib import Path
from .io import write,save,sha
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/a1_object_v1'


def config(seed,run_root=None):
    assert seed in (42,43,44)
    source=ROOT/('runs/own_reference_control/ifa/seed%d/run_001/config.json'%seed)
    cfg=json.loads(source.read_text());assert cfg['ifa_own_reference_fix'] is True
    run=Path(run_root) if run_root else ROOT/('runs/a1_object_v1/seed%d'%seed)
    cfg.update(pretrain=str(ROOT/'checkpoints/object_multitask_model_ep99.pth'),
        feature_dir=str(ROOT/('features/object_seed%d'%seed)),buffer_dir=str(run/'buffer'),
        experiment_dir=str(run/'checkpoints'),experiment_root=str(run),folder=str(ROOT/'data'),
        bddl_folder=str(ROOT/'libero/libero/bddl_files'),init_states_folder=str(ROOT/'libero/libero/init_files'),
        a1_root=str(run),a1_report=str(OUT/('seed%d'%seed)),a1_recovery=str(run/'recovery/latest.pt'),
        a1_protocol=str(OUT/'PROTOCOL.json'),a1_base_reference=str(OUT/'PRETRAINED_REFERENCES.pt'),a1_enabled=True,
        ifa_own_reference_fix=True,ifa_interaction_logging=True,ifa_interaction_log_dir=str(run/'ifa_interaction'),
        use_wandb=False,repro_arm='A1_OBJECT_REFERENCE_SUBSPACE',training=True)
    for k in ('e1','p1','g1','r1','a2','a3','tsr'):cfg[k+'_enabled']=False
    return cfg


def code_hashes():
    files=list((ROOT/'a1_object_v1').glob('*.py'))+list((ROOT/'a1_object_v1').glob('*.sh'))
    native=json.loads((ROOT/'reports/v1_object_tsr/EXPERIMENT_CONTEXT.json').read_text())['frozen_source_files']
    files += [ROOT/p for p in native if p.startswith('libero/')]
    files += [ROOT/'libero/lifelong/models/complete_policy.py',ROOT/'libero/lifelong/models/modules/language_modules.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['inputs','freeze'],required=True);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    from easydict import EasyDict
    import torch
    if a.phase=='inputs':
        from .metrics import exact_metrics,floats
        from .references import recover_base
        source=ROOT/'reports/own_reference_control/CONTROLLED_RESULTS.json'
        data=json.loads(source.read_text())['results']['corrected_ifa']['per_seed']
        base={s:dict(matrix=r['r'],metrics=floats(exact_metrics(r['r'])),
            exact_metrics={k:str(v) for k,v in exact_metrics(r['r']).items()}) for s,r in data.items()}
        write(OUT/'BASELINE_REFERENCE.json',dict(source=str(source),sha256=sha(source),results=base,
            baseline_commit='d75fb371cf575beb079ce064d70d2ae323da95dd',historical_gpu='RTX4090',current_gpu=torch.cuda.get_device_name(0),
            cross_environment_limit='Historical full-trajectory bitwise identity not claimed; no outcome-matching reruns.'))
        for seed in (42,43,44):write(OUT/('CONFIG_seed%d.json'%seed),config(seed))
        if not (OUT/'PRETRAINED_REFERENCES.pt').exists():
            state=recover_base(EasyDict(config(42)));save(OUT/'PRETRAINED_REFERENCES.pt',state)
            write(OUT/'REFERENCE_MANIFEST.json',state['metadata'])
        write(OUT/'VALUE_ASSESSMENT.json',dict(decision='EXECUTE_BOUNDED_SEED42_SCREEN',
            basis='Verified corrected-reference baseline; distinct unresolved IFA normalization/complement-space mechanism.',
            limits=['Not a shared-skill proof','Projection changes gradient sensitivity as well as its feature-space support',
                'Historical comparator GPU differs; same-state parity does not prove identical training trajectories',
                'Single-seed positive delta only authorizes expansion, not significance'],
            additional_training_initially='one OBJECT seed42 four-stage run',evaluation_trials_initially=200))
    else:
        assert not (OUT/'PROTOCOL.json').exists()
        check=json.loads((OUT/'ENGINEERING_CHECK.json').read_text());assert check['status']=='PASS'
        assert check['source_files']==code_hashes(),'CODE_CHANGED_AFTER_SMOKE'
        cfg=config(42)
        write(OUT/'PROTOCOL.json',dict(name='A1-OBJECT-v1',user_protocol_sha256=sha(ROOT/'a1_object_v1/USER_PROTOCOL.md'),
            own_reference_fix=True,baseline_reference_sha256=sha(OUT/'BASELINE_REFERENCE.json'),
            initial_references_sha256=sha(OUT/'PRETRAINED_REFERENCES.pt'),implementation_files=code_hashes(),
            tasks=[6,7,8,9],epochs_per_stage=100,primary_trials_per_seed=200,initial_seed=42,
            seed_order=[42,43,44],screen='unrounded AUC seed42 > corrected baseline; then 43 and 44',
            expansion='Positive three-seed mean then GOAL/LIBERO50 only with verified same-reference baselines; never synthesize missing inputs',
            only_algorithm_change='IFA raw last_feat -> (last_feat@U)@U.T before first native normalization; policy head unchanged',
            reference_scope='pretrained tasks plus arrived current tasks',basis='FP64 CPU two-pass append-only MGS',
            rank_residual=1e-7,orth64=1e-10,reconstruction64=1e-6,validation32=1e-5,small_rho_log_only=1e-6,
            projection_dtype='float32',extra_parameter_gradient_probes=False,geometry_every_steps=100,
            baseline_config_sha256={str(s):sha(ROOT/('runs/own_reference_control/ifa/seed%d/run_001/config.json'%s)) for s in (42,43,44)},
            run_config_sha256={str(s):sha(OUT/('CONFIG_seed%d.json'%s)) for s in (42,43,44)},
            pretrain_sha256=sha(cfg['pretrain']),baseline_training_rerun=False,pretraining_rerun=False,
            old_experiment_queues_resumed=False))
        write(OUT/'EXPANSION_ASSET_INVENTORY.json',dict(checked_scope='Existing local checkpoints, feature roots and corrected-reference control report',
            checkpoint_files=sorted(p.name for p in (ROOT/'checkpoints').iterdir()),
            feature_roots=sorted(p.name for p in (ROOT/'features').iterdir()),
            corrected_baseline_scope='LIBERO_OBJECT only',
            goal_and_libero50_ready=False,
            missing='GOAL/LIBERO50 pretrained checkpoints, feature caches and corrected-reference baseline matrices are not present in the canonical reused asset roots.',
            consequence='Only if OBJECT three-seed mean improves, obtain and verify existing external expansion assets; do not train replacement baselines.'))
        write(OUT/'EXPERIMENT_CONTEXT.json',dict(server_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
            dirty=subprocess.check_output(['git','status','--short','--untracked-files=no'],text=True),
            python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
            source_files=code_hashes(),formal_old_experiment_queues_resumed=False))


if __name__=='__main__':main()
