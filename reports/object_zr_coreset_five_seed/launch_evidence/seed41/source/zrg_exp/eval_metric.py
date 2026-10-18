"""Exact baseline rollout function, plus per-trial recording only."""
from libero.lifelong.metric import *
import gc
from .evaluate import record_trial

def evaluate_one_task_success(
    cfg, algo, task, instr, mask, task_id, sim_states=None, task_str="", render = True,see=None
):
    """
    Evaluate a single task's success rate
    sim_states: if not None, will keep track of all simulated states during
                evaluation, mainly for visualization and debugging purpose
    task_str:   the key to access sim_states dictionary
    """
    # render = cfg.eval.render
    with Timer() as t:
        if cfg.lifelong.algo == "PackNet":  # need preprocess weights for PackNet
            algo = algo.get_eval_algo(task_id)

        algo.eval()
        env_num =  10
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # initiate evaluation envs
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        # Try to handle the frame buffer issue

        from zrg_exp.evaluate import ledger
        ledger('environment_construct_start',task=task_id,lanes=env_num)
        env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
        )

        ledger('environment_construct_complete',task=task_id,lanes=env_num)
        ### Evaluation loop
        # get fixed init states to control the experiment randomness
        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        ####################################################################################################################
        num_success = 0
        for i in range(eval_loop_num):
            seed = see + i
            ledger('vector_attempt_start',task=task_id,group=i,env_seed=seed,initial_indices=list(range(i*env_num,(i+1)*env_num)))
            env.seed(seed)
            env.reset()
            env.seed(seed) 
            indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
            init_states_ = init_states[indices]

            dones = [False] * env_num
            steps = 0
            algo.reset()
            obs = env.set_init_state(init_states_)
            history_buffer = []
           
            # dummy actions [env_num, 7] all zeros for initial physics simulation
            dummy = np.zeros((env_num, 7))
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            frames = []
            with torch.no_grad():

                while steps < cfg.eval.max_steps:
                    steps += 1

                    data = raw_obs_to_tensor_obs(obs, instr, mask, cfg)

                    history_buffer.append(data)
                    if len(history_buffer) > 8:
                        history_buffer.pop(0)

                    sliding_window = history_buffer.copy()


                    def stack_trajectory(data_buffer):
                        assert len(data_buffer) > 0
                        result = {
                            "obs": {},
                            "task_emb": data_buffer[0]["task_emb"],  # 不变
                            "masks": data_buffer[0]["masks"],  # 不变
                        }
                        obs_keys = data_buffer[0]["obs"].keys()
                        for key in obs_keys:
                            result["obs"][key] = torch.stack(
                                [step["obs"][key] for step in data_buffer],
                                dim=1  # 这里就是 B,T,C,H,W
                            )
                        return result

                    batch_data = stack_trajectory(sliding_window)

                    actions = algo.policy.get_action(batch_data)

                    obs, reward, done, info = env.step(actions)

                    # check whether succeed
                    for k in range(env_num):
                        dones[k] = dones[k] or done[k]

                    if all(dones):
                        break

                # a new form of success record
                for k in range(env_num):
                    if i * env_num + k < cfg.eval.n_eval:
                        record_trial(cfg, task_id, see, i * env_num + k, int(indices[k]), bool(dones[k]), steps)
                        num_success += int(dones[k])

            ledger('vector_attempt_complete',task=task_id,group=i,env_seed=seed,steps=steps,successes=[bool(x) for x in dones],lanes=env_num)
        success_rate = num_success / cfg.eval.n_eval
        env.close()
        ledger('environment_closed',task=task_id,lanes=env_num)
        gc.collect()
    print(f"[info] evaluate task {task_id} takes {t.get_elapsed_time():.1f} seconds")
    return success_rate
