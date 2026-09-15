import subprocess,time,sys,platform
from .common import *
def run():
    assert not OUT.exists();OUT.mkdir(parents=True)
    old=read(PRIOR/'PROTOCOL_FROZEN.json')
    assert read(PRIOR/'INDEPENDENT_AUDIT.json')['status']=='PASS'
    from cb_ifa_experiment.gpu_runtime import eval_config,torch,digest
    assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
    cfgs={str(s):eval_config(s,OUT) for s in old['seeds']}
    normalized=[]
    for s,c in cfgs.items():
        assert c.eval.n_eval==20 and c.eval.max_steps==600
        assert c.policy.use_language is True and c.policy.use_state is False
        normalized.append(digest([dict(c.policy),dict(c.data),dict(c.eval)]))
        write(OUT/('EVAL_CONFIG_seed'+s+'.json'),c)
    assert len(set(normalized))==1
    tasks={n:list(range(6,int(n.split('_')[1][1:])+1)) for n in old['points']}
    cells=[]
    for i,n in enumerate(old['points']):
        order=ARMS if i%2==0 else ARMS[::-1]
        for task in tasks[n]:
            for arm in order:cells.append({'point':n,'point_index':i,'task':task,'arm':arm,'environment_seed':int(n.split('_')[0][1:]),'round_policy_seeds':[round_seed(i,task,r) for r in range(2)]})
    assert len(cells)==180
    p={'name':'paper_qp_rollout_v1','frozen_at':time.time(),'points':old['points'],'preflight6':old['preflight6'],'seeds':old['seeds'],'arms':ARMS,
       'endpoint':20,'training':'Faithful replay ONLY of original20 updates, no new trajectory, extension, hyperparameter or method change. Both arms must reproduce prior full endpoint witnesses at1/5/20 bitwise.',
       'evaluation':'Native evaluate_one_task_success,600 policy steps maximum,10 parallel lanes,two rounds,20 fixed trials per cell; original get_action GMM sample and success OR, five zero-action warmup steps, history8.',
       'evaluation_randomness':'Reset Python/NumPy/torchCPU/CUDA at each round set_init_state to predetermined round_policy_seed, shared by arms. This explicitly differs from historical cross-task/cross-round RNG carry; no changes to action sampling law. Environment seed=training seed+round, native state indices0..19.',
       'primary':'Within each origin, equal mean of seen task6..current successes; then equal origins within each seed; then equal3seeds. Report QP-minus-MLR percentage points.',
       'secondary':['old task balanced success','current task success','per task and seed','worst old task per origin','phase','paired trial wins/losses/ties'],
       'tasks_by_point':tasks,'cells':cells,'trials_per_cell':20,'total_primary_trials':3600,
       'engineering':'6fixed recovery points; identical-model duplicate complete20trials on first fixed origin/task6 with passive trace on/off, exact actions/dones/initial-state hashes/RNG; no effect-based tuning. Engineering trials excluded.',
       'decision':{'candidate_for_full_training':'mean primary >=2.5pp, >=2seed primary positive, no seed primary <=-5pp; pooled old/new >=-2.5pp and each seed old/new >-5pp; no evaluated task degrades >=10pp in>=2seeds',
                   'not_supported':'pooled primary <=0pp','otherwise':'INCONCLUSIVE','threshold_status':'new development thresholds frozen before any rollout outcome; not significance guarantees'},
       'uncertainty':'36related origins,3development seeds and repeated initial states are NOT3600independent experiments. Seed effects displayed. Stratified origin bootstrap10000 private CPU draws gives conditional descriptive interval only, not population-seed inference.',
       'limitations':'Method chosen using BC on reused development origins. No new held-out seed confirmation. Endpoints20steps, not complete lifelong training; task9 not in these origins. No BC substitution for missing rollout.',
       'resume':'Completed cells hash-verified and preserved. Incomplete failed cell may be faithfully retried only for technical failure with failure retained; no score-based reruns or trial additions.',
       'resources':{'one_GPU_worker':True,'GPU':'NVIDIA GeForce RTX 3090','data_free_min_GiB':1,'root_free_min_GiB':8,'wall_watchdog_hours':72,'completed_cells_never_overwritten':True}}
    write(OUT/'PROTOCOL_FROZEN.json',p)
    files=set()
    for source in [PRIOR/'FROZEN_INPUTS.json']:
        for rel,h in read(source).items():assert sha(ROOT/rel)==h,rel;files.add(ROOT/rel)
        files.add(source)
    files.update((ROOT/'paper_qp_rollout_v1').glob('*.py'))
    files.update((ROOT/'libero/lifelong').rglob('*.py'))
    files.update((ROOT/'libero/configs').rglob('*.yaml'))
    files.update((ROOT/'libero/libero/envs').rglob('*.py'))
    files.update((ROOT/'libero/libero/bddl_files/libero_object').glob('*'))
    files.update((ROOT/'libero/libero/init_files/libero_object').glob('*'))
    files.add(ROOT/'checkpoints/object_multitask_model_ep99.pth')
    files.update(OUT.glob('EVAL_CONFIG*.json'));files.add(OUT/'PROTOCOL_FROZEN.json')
    for n in p['points']:
        files.add(PRIOR/'measurements'/n/'ENDPOINT_STATES.jsonl');files.add(PRIOR/'measurements'/n/'DECISIONS_FROZEN.jsonl')
    write(OUT/'FROZEN_INPUTS.json',{x.relative_to(ROOT).as_posix():sha(x) for x in sorted(files) if x.is_file()})
    write(OUT/'ENVIRONMENT.json',{'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'git_status':subprocess.check_output(['git','status','--short'],text=True),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)})
    write(OUT/'ASSET_AUDIT.json',{'status':'INPUT_HASHES_PASS','points':36,'exported_endpoints':'PENDING','complete_state_parity':'required before rollout','historical_assets_overwritten':False})
    print('PROTOCOL_FROZEN',sha(OUT/'PROTOCOL_FROZEN.json'),flush=True)
if __name__=='__main__':run()
