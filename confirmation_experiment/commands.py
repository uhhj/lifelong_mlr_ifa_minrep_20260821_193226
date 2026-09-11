"""Commands preserve the native Hydra recipes and entry points."""
import sys
from confirmation_experiment.common import ROOT,RUNS

def feature_command(seed):
    return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
            'libero/lifelong/save_feature.py',f'seed={seed}','benchmark_name=LIBERO_OBJECT',
            'policy=complete_policy','lifelong=basic','folder='+str(ROOT/'data'),
            'bddl_folder='+str(ROOT/'libero/libero/bddl_files'),
            'init_states_folder='+str(ROOT/'libero/libero/init_files'),
            'train.batch_size=1','train.num_workers=0','pretrain='+str(ROOT/'checkpoints/object_multitask_model_ep99.pth'),
            '+start_task=6','+end_task=10','+aug=true','+feature_dir='+str(ROOT/f'features/object_seed{seed}'),
            '+experiment_root='+str(RUNS/f'feature_seed{seed}'),'training=false']

def train_command(seed,arm,smoke=False):
    base=RUNS/('smoke' if smoke else 'formal')/f'seed{seed}'/arm
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
         '--module','confirmation_experiment.train_entry',f'seed={seed}','benchmark_name=LIBERO_OBJECT',
         'policy=BriefPolicy','lifelong='+('mlr' if arm=='mlr' else 'mlr_fixed_kappa'),
         'lifelong.algo='+('ConfirmationMLR' if arm=='mlr' else 'ConfirmationSoft'),
         'folder='+str(ROOT/'data'),'pretrain='+str(ROOT/'checkpoints/object_multitask_model_ep99.pth'),
         'train.n_epochs='+('1' if smoke else '100'),'train.batch_size=10','train.num_workers=0',
         'train.optimizer.kwargs.lr=1e-4','train.optimizer.kwargs.betas=[0.9,0.999]',
         'train.optimizer.kwargs.eps=1e-8','train.optimizer.kwargs.weight_decay=0.1',
         '+start_task=6','+end_task='+('8' if smoke else '10'),'+aug=false',
         '+feature_dir='+str(ROOT/f'features/object_seed{seed}'),
         '+buffer_dir='+str(ROOT/'buffers/fixed_kappa_confirmation_v2'/('smoke' if smoke else 'formal')/f'seed{seed}'/arm),
         '+experiment_root='+str(base),'+past_data_rate=0.5','+ifa_enabled=false','+ifa_weight=0.1','+gamma=0.30',
         '+repro_scheduler_mode=batch_step','+repro_arm=confirmation_v2_'+arm,
         '+save_final_checkpoint=true','training=true']
    if arm=='soft':cmd+=['lifelong.kappa=0.5','lifelong.projection_enabled=true','lifelong.step_log_every=0']
    if smoke:cmd+=['+max_optimizer_updates=8']
    return cmd

def eval_command(seed,arm):
    run=RUNS/'formal'/f'seed{seed}'/arm
    return [sys.executable,'-m','confirmation_experiment.eval_entry',f'seed={seed}',
            'benchmark_name=LIBERO_OBJECT','policy=complete_policy','lifelong=basic',
            'folder='+str(ROOT/'data'),'pretrain='+str(ROOT/'checkpoints/object_multitask_model_ep99.pth'),
            'bddl_folder='+str(ROOT/'libero/libero/bddl_files'),
            'init_states_folder='+str(ROOT/'libero/libero/init_files'),
            'eval.n_eval=20','eval.num_procs=20','+start_task=6','+end_task=10','+aug=true',
            '+epoch_dir='+str(run/'run_001'),'+test_log_file='+str(run/'primary_ep99.log'),
            '+eval_epochs=[99]','+experiment_root='+str(run/'eval_primary_ep99'),'training=false']
