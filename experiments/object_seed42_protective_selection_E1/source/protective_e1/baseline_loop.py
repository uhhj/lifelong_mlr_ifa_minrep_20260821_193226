# Baseline MLR loop; only epoch range changed for recovery.
# Source SHA256: 97f8ebb8ffdd54f5596c956c8f125feac70ebdbd913386bc8384c58d4a7463e5
from libero.lifelong.algos.mlr import *
from libero.lifelong.algos.mlr import _is_rank0

class E1EpochLoop(MLR):
    def learn_all_tasks(self, datasets, benchmark, result_summary,current_task_id):

        ifa_info = None
        if getattr(self.cfg, "ifa_enabled", True):
            ifa_info = build_ifa_similarity_matrix(
                current_task_id=current_task_id,
                current_feat_root=self.cfg.feature_dir,
                replay_dir=self.cfg.buffer_dir,
                language_idx=0,
                agent_idx=1,
                top_ratio=0.5,
            )

        selected_pairs = [] if ifa_info is None else ifa_info["selected_pairs"]
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(
            f"[protocol] seed={self.cfg.seed} gpu={gpu_name} "
            f"stage={current_task_id} ifa_enabled={getattr(self.cfg, 'ifa_enabled', True)} "
            f"selected_pairs={selected_pairs}"
        )

        base_dir = self.cfg.feature_dir
        buffer_dir = self.cfg.buffer_dir

        dataset = build_dataset_for_task(base_dir, current_task_id, buffer_dir, self.cfg.seed, start=self.cfg.start_task)
        print("Train set size:", len(dataset))

        # learn on all tasks, only used in multitask learning

        train_sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=train_sampler,
            persistent_workers=False,
        )
        self.start_task(-1,train_dataloader)

        cumulated_counter = 0.0
        successes = []
        losses = []

        # start training
        for epoch in range(self.e1_start_epoch, self.cfg.train.n_epochs):
            train_sampler.set_epoch(epoch)
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", unit="batch")
            t0 = time.time()
            self.policy.train()
            training_loss = 0.0
            epoch_loss_sum = 0.0
            epoch_batch_count = 0
            optimizer_step_start = self._optimizer_step_count
            lr_start = float(self.optimizer.param_groups[0]["lr"])
            lr_values = [lr_start]
            interaction_logging = bool(
                getattr(self.cfg, "ifa_interaction_logging", False)
            )
            interaction_sums = {
                "base": 0.0,
                "hinge": 0.0,
                "active": 0.0,
                "added": 0.0,
                "current_samples": 0,
            }
            for (idx, data) in enumerate(pbar):
                loss = self.observe(data,current_task_id,epoch,ifa_info=ifa_info)
                if interaction_logging:
                    stats = self._last_ifa_interaction_stats
                    interaction_sums["base"] += stats["base_bc_loss_before_ifa"]
                    current_count = stats["current_task_sample_count"]
                    interaction_sums["hinge"] += stats["ifa_hinge_mean"] * current_count
                    interaction_sums["active"] += stats["ifa_hinge_active_fraction"] * current_count
                    interaction_sums["added"] += stats["ifa_added_loss"]
                    interaction_sums["current_samples"] += current_count
                epoch_loss_sum += float(loss)
                epoch_batch_count += 1
                training_loss += loss
                avg_loss = training_loss / (idx + 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
                lr_values.append(float(self.optimizer.param_groups[0]["lr"]))

                training_loss /= len(train_dataloader)

                max_updates = getattr(self.cfg, "max_optimizer_updates", None)
                if (
                    max_updates is not None
                    and self._optimizer_step_count >= int(max_updates)
                ):
                    break
                
            t1 = time.time()

            current_lr = self.optimizer.param_groups[0]["lr"]
            mean_bc_loss = epoch_loss_sum / epoch_batch_count
            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} "
                f"| lr: {current_lr:.8g} | time: {(t1-t0)/60:4.2f}"
            )

            reached_max_updates = (
                getattr(self.cfg, "max_optimizer_updates", None) is not None
                and self._optimizer_step_count >= int(self.cfg.max_optimizer_updates)
            )
            save_final_checkpoint = getattr(self.cfg, "save_final_checkpoint", False)
            diagnostic_epochs = set(
                int(item)
                for item in getattr(self.cfg, "diagnostic_checkpoint_epochs", [])
            )
            if (
                should_save_mlr_checkpoint(
                    epoch,
                    self.cfg.train.n_epochs,
                    diagnostic_epochs,
                    save_final_checkpoint,
                )
                or (save_final_checkpoint and reached_max_updates)
            ):

                self.policy.eval()

                model_checkpoint_name_ep = os.path.join(
                    self.experiment_dir, f"mlrifa_{current_task_id}_model_ep{epoch}.pth"
                )
                if not dist.is_initialized() or dist.get_rank() == 0:

                    torch_save_model(self.policy, model_checkpoint_name_ep, cfg=self.cfg)
                    print(
                        f"[checkpoint] seed={self.cfg.seed} stage={current_task_id} "
                        f"epoch={epoch} path={model_checkpoint_name_ep}"
                    )
                    losses.append(training_loss)

                # Defaultly, we don't do evaluation during training.
                if self.cfg.lifelong.eval_in_train:
                    success_rates = evaluate_multitask_training_success(
                        self.cfg, self, benchmark, all_tasks
                    )
                    success_rate = np.mean(success_rates)
                else:
                    success_rate = 0.0
                successes.append(success_rate)
                cumulated_counter += 1.0

            self.epoch_schedule_step(epoch)

            arm = str(getattr(self.cfg, "repro_arm", "unspecified"))
            self._append_csv(
                os.path.join(self.experiment_dir, "lr_protocol.csv"),
                LR_PROTOCOL_FIELDS,
                {
                    "seed": self.cfg.seed,
                    "arm": arm,
                    "stage": current_task_id,
                    "epoch": epoch,
                    "batches": epoch_batch_count,
                    "optimizer_step_start": optimizer_step_start,
                    "optimizer_step_end": self._optimizer_step_count,
                    "lr_start": lr_start,
                    "lr_end": float(self.optimizer.param_groups[0]["lr"]),
                    "lr_min": min(lr_values),
                    "lr_max": max(lr_values),
                    "scheduler_mode": self.repro_scheduler_mode,
                    "batch_size": self.cfg.train.batch_size,
                    "warmup_steps": self.scheduler_warmup_steps,
                    "total_steps": self.scheduler_total_steps,
                },
            )

            if interaction_logging and _is_rank0():
                interaction_log_dir = str(
                    getattr(
                        self.cfg,
                        "ifa_interaction_log_dir",
                        os.path.join(self.experiment_dir, "ifa_interaction"),
                    )
                )
                os.makedirs(interaction_log_dir, exist_ok=True)
                mean_base = interaction_sums["base"] / epoch_batch_count
                current_samples = interaction_sums["current_samples"]
                mean_hinge = interaction_sums["hinge"] / max(current_samples, 1)
                mean_added = interaction_sums["added"] / epoch_batch_count
                self._append_csv(
                    os.path.join(
                        interaction_log_dir,
                        f"seed{self.cfg.seed}_ifa_epoch_stats.csv",
                    ),
                    IFA_INTERACTION_FIELDS,
                    {
                        "seed": self.cfg.seed,
                        "stage": current_task_id,
                        "epoch": epoch,
                        "batch_count": epoch_batch_count,
                        "current_sample_count": interaction_sums["current_samples"],
                        "mean_base_bc_loss": mean_base,
                        "mean_ifa_hinge": mean_hinge,
                        "ifa_active_fraction": interaction_sums["active"] / max(current_samples, 1),
                        "mean_ifa_added_loss": mean_added,
                        "ifa_to_bc_ratio": mean_added / max(abs(mean_base), 1e-12),
                        "lr_start": lr_start,
                        "lr_end": float(self.optimizer.param_groups[0]["lr"]),
                        "optimizer_steps": self._optimizer_step_count - optimizer_step_start,
                        "selected_pairs": json.dumps(selected_pairs),
                    },
                )

            self._append_csv(
                os.path.join(self.experiment_dir, "paper_repro_train_curve.csv"),
                TRAIN_CURVE_FIELDS,
                {
                    "seed": self.cfg.seed,
                    "arm": arm,
                    "stage": current_task_id,
                    "epoch": epoch,
                    "batches": epoch_batch_count,
                    "mean_bc_loss": mean_bc_loss,
                    "optimizer_step_end": self._optimizer_step_count,
                    "lr_end": float(self.optimizer.param_groups[0]["lr"]),
                    "scheduler_mode": self.repro_scheduler_mode,
                    "batch_size": self.cfg.train.batch_size,
                },
            )

            max_updates = getattr(self.cfg, "max_optimizer_updates", None)
            if (
                max_updates is not None
                and self._optimizer_step_count >= int(max_updates)
            ):
                break

        # return the metrics regarding forward transfer
        losses = np.array(losses)
        successes = np.array(successes)

        total, added, dropped, quota, counts = update_buffer_equal_quota_10(
            base_dir, current_task_id, buffer_dir,
            capacity=BUFFER_CAPACITY, seed= self.cfg.seed, drop_policy="random",rate=self.cfg.past_data_rate
        )
        return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter
