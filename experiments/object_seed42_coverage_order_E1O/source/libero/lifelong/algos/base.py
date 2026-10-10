import os
import time

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from libero.lifelong.metric import *
from libero.lifelong.models import *
from libero.lifelong.utils import *
from libero.lifelong.utils import make_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import math

import pandas as pd
import itertools
REGISTERED_ALGOS = {}

def hinge_weight_sigmoid(epoch, t0=78, alpha=3.0, w_max=0.1):
    s = 1.0 / (1.0 +  math.exp(-(epoch - t0)))
    return w_max * s

def l2norm(x, eps=1e-8):
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

def register_algo(policy_class):
    """Register a policy class with the registry."""
    policy_name = policy_class.__name__.lower()
    if policy_name in REGISTERED_ALGOS:
        raise ValueError("Cannot register duplicate policy ({})".format(policy_name))

    REGISTERED_ALGOS[policy_name] = policy_class


def get_algo_class(algo_name):
    """Get the policy class from the registry."""
    if algo_name.lower() not in REGISTERED_ALGOS:
        raise ValueError(
            "Policy class with name {} not found in registry".format(algo_name)
        )
    return REGISTERED_ALGOS[algo_name.lower()]


def get_algo_list():
    return REGISTERED_ALGOS


class AlgoMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all algorithms that should not be registered here.
        _unregistered_algos = []

        if cls.__name__ not in _unregistered_algos:
            register_algo(cls)
        return cls


class Sequential(nn.Module, metaclass=AlgoMeta):
    """
    The sequential finetuning BC baseline, also the superclass of all lifelong
    learning algorithms.
    """

    def __init__(self, n_tasks, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss_scale = cfg.train.loss_scale
        self.n_tasks = n_tasks
        if not hasattr(cfg, "experiment_dir"):
            create_experiment_dir(cfg)
            print(
                f"[info] Experiment directory not specified. Creating a default one: {cfg.experiment_dir}"
            )
        self.experiment_dir = cfg.experiment_dir
        self.algo = cfg.lifelong.algo

        self.policy = get_policy_class(cfg.policy.policy_type)(cfg, cfg.shape_meta)
        self.current_task = -1

    def end_task(self, dataset, task_id, benchmark, env=None):
        """
        What the algorithm does at the end of learning each lifelong task.
        """
        pass

    def start_task(self, task, train_loader):
        """
        What the algorithm does at the beginning of learning each lifelong task.
        """
        self.current_task = task

        self.optimizer = eval(self.cfg.train.optimizer.name)(
            self.policy.parameters(),
            **self.cfg.train.optimizer.kwargs
        )
        self.repro_scheduler_mode = str(
            getattr(self.cfg, "repro_scheduler_mode", "legacy_epoch")
        )
        valid_scheduler_modes = {"legacy_epoch", "batch_step", "constant"}
        if self.repro_scheduler_mode not in valid_scheduler_modes:
            raise ValueError(
                f"Unknown repro_scheduler_mode={self.repro_scheduler_mode!r}; "
                f"expected one of {sorted(valid_scheduler_modes)}"
            )
        self._optimizer_step_count = 0
        self._scheduler_step_count = 0
        policy_module = self.policy.module if isinstance(self.policy, DDP) else self.policy
        for name, param in policy_module.named_parameters():
            if param.requires_grad:
                print(name)

        # build scheduler
        self.scheduler = None
        self.scheduler_warmup_steps = None
        self.scheduler_total_steps = None
        if (
            self.cfg.train.scheduler is not None
            and self.repro_scheduler_mode != "constant"
        ):
            sched_cfg = self.cfg.train.scheduler.kwargs
            warmup_steps = sched_cfg.get("warmup_steps", 500)

            total_steps  = getattr(self, "total_steps", None)
            if total_steps is None:
                total_steps = self.cfg.train.n_epochs * len(train_loader)

            lr_lambda = make_linear_schedule_with_warmup(warmup_steps, total_steps)

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lr_lambda,
                last_epoch=sched_cfg.get("last_epoch", -1),
            )
            self.scheduler_warmup_steps = int(warmup_steps)
            self.scheduler_total_steps = int(total_steps)

    def optimizer_step_and_maybe_schedule(self):
        """Apply one optimizer update and, for batch mode, one scheduler update."""
        self.optimizer.step()
        self._optimizer_step_count += 1
        if (
            self.scheduler is not None
            and self.repro_scheduler_mode == "batch_step"
        ):
            self.scheduler.step()
            self._scheduler_step_count += 1

    def epoch_schedule_step(self, epoch):
        """Preserve the release's epoch stepping only in legacy mode."""
        if (
            self.scheduler is not None
            and self.repro_scheduler_mode == "legacy_epoch"
            and epoch > 0
        ):
            self.scheduler.step()
            self._scheduler_step_count += 1

    def map_tensor_to_device(self, data):
        """Move data to the device specified by self.cfg.device."""
        return TensorUtils.map_tensor(
            data, lambda x: safe_device(x, device=self.cfg.device)
        )

    def observe(self, data,current_task_id,epoch,ifa_info=None):
        """
        How the algorithm learns on each data point -- IFA.
        """
        data[0] = self.map_tensor_to_device(data[0])
        self.optimizer.zero_grad()
        if isinstance(self.policy, DDP):
            loss,last_feat = self.policy.module.compute_loss(data[0],None,None)
        else:
            loss,last_feat = self.policy.compute_loss(data[0],None,None)
        
        print('loss:{}'.format(loss))
        interaction_logging = bool(
            getattr(self.cfg, "ifa_interaction_logging", False)
        )
        base_bc_loss = loss.detach() if interaction_logging else None
        logged_hinge_values = []
        logged_old_tasks = []

        task_ids = torch.tensor(
            [int(x) for x in data[1]], dtype=torch.long, device=last_feat.device
        )

        if not getattr(self.cfg, "ifa_enabled", True) or ifa_info is None:
            (self.loss_scale * loss).backward()
            if self.cfg.train.grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.train.grad_clip
                )
            self.optimizer_step_and_maybe_schedule()
            return loss.item()

        selected_pair_mask = ifa_info["selected_pair_mask"]
        sim_language = ifa_info["sim_language"]
        sim_agent = ifa_info["sim_agent"]

        ifa_task_ids = ifa_info["task_ids"]

        if not torch.is_tensor(selected_pair_mask):
            selected_pair_mask = torch.as_tensor(selected_pair_mask)

        pairs_below_num = []

        for i, j in itertools.combinations(range(len(ifa_task_ids)), 2):
            if selected_pair_mask[i, j].item():
                value = sim_language[i, j].item()

                pairs_below_num.append(
                    (
                        int(ifa_task_ids[i]),
                        int(ifa_task_ids[j]),
                        value,
                    )
                )

        batch_mask = (task_ids == current_task_id) & (task_ids != 6)

        if ifa_info is not None and batch_mask.any().item() and current_task_id > 6:

            batch_size = batch_mask.sum().item()
            true_feats = last_feat[batch_mask]

            cur_feat = data[0]["feature"]
            cur_feat = cur_feat.to(true_feats.device).float()

            # [B, 1, time, modality, E] -> [B, time, modality, E]
            if cur_feat.dim() == 5 and cur_feat.shape[1] == 1:
                cur_feat = cur_feat.squeeze(1)

            sample = cur_feat[0, 0, 0, :]   # [E], historical mixed-batch first reference
            own_references = (
                cur_feat[batch_mask, 0, 0, :]
                if getattr(self.cfg, "ifa_own_reference_fix", False)
                else None
            )

            task_id_list = ifa_info["task_ids"]
            selected_pair_mask = ifa_info["selected_pair_mask"]

            if not torch.is_tensor(selected_pair_mask):
                selected_pair_mask = torch.as_tensor(selected_pair_mask)

            task_id_tensor = torch.as_tensor(task_id_list)

            cur_pos = (task_id_tensor == current_task_id).nonzero(as_tuple=True)[0]

            if len(cur_pos) == 0:
                low_sim_tasks = []
            else:
                cur_idx = cur_pos.item()

                true_indices = selected_pair_mask[cur_idx].nonzero(as_tuple=True)[0]

                low_sim_tasks = [
                    int(task_id_tensor[idx].item())
                    for idx in true_indices
                    if int(task_id_tensor[idx].item()) != current_task_id
                ]

            print(f"[IFA] current_task_id={current_task_id}, low_sim_tasks={low_sim_tasks}")
            if  len(low_sim_tasks)==0:
                loss_hinge = None

            else:
                for item in low_sim_tasks:
                    true_lan_list = []
                    flase_lan_list = []
                    for i in range(batch_size):
                        true_lan_list.append(sample if own_references is None else own_references[i])
                        false_lan_name = f'{self.cfg.feature_dir}/{item}/0.pt'
                        flase_lan_list.append(
                            torch.load(false_lan_name, map_location=true_feats.device)["feature"][0, 0, 0, :]
                        )

                    true_lan_list = torch.stack(true_lan_list, dim=0)  # (B,E)
                    false_lan_list = torch.stack(flase_lan_list, dim=0)  # (B, E)

                    true_feats  = l2norm(true_feats)          # (B,E)
                    true_lan_list = l2norm(true_lan_list)      # (B,E)
                    false_lan_list = l2norm(false_lan_list)     # (B,E)

                    sim_language = ifa_info["sim_language"]
                    ifa_task_ids = ifa_info["task_ids"]

                    if torch.is_tensor(sim_language):
                        sim_language_cpu = sim_language.detach().cpu()
                    else:
                        sim_language_cpu = torch.as_tensor(sim_language)

                    sim = {}

                    for i, tid_i in enumerate(ifa_task_ids):
                        tid_i = int(tid_i)
                        sim[tid_i] = {}

                        for j, tid_j in enumerate(ifa_task_ids):
                            tid_j = int(tid_j)
                            sim[tid_i][tid_j] = float(sim_language_cpu[i, j].item())

                    gamma = self.cfg.gamma
                    eps = 1e-6

                    true_feats       = l2norm(true_feats)          # (B,E)
                    true_lan_list   = l2norm(true_lan_list)       # (B,E)
                    false_lan_list  = l2norm(false_lan_list)      # (B,E)

                    cos_pos = (true_feats * true_lan_list).sum(-1).clamp(-1 + eps, 1 - eps)   # (B,)
                    cos_neg = (true_feats * false_lan_list).sum(-1).clamp(-1 + eps, 1 - eps)  # (B,)

                    theta_pos = torch.acos(cos_pos)  # (B,)
                    theta_neg = torch.acos(cos_neg)  # (B,)

                    cos_ab = sim[current_task_id][item]
                    cos_ab_t = torch.tensor(cos_ab, dtype=true_feats.dtype, device=true_feats.device).clamp(-1 + eps, 1 - eps)
                    phi = torch.acos(cos_ab_t)

                    hinge_values = torch.relu(theta_pos - theta_neg + gamma * phi)
                    loss_hinge_angle = hinge_values.mean()

                    ifa_weight = float(getattr(self.cfg, "ifa_weight", 0.1))
                    loss = loss + ifa_weight * loss_hinge_angle.mean()
                    if interaction_logging:
                        logged_hinge_values.append(hinge_values.detach())
                        logged_old_tasks.append(int(item))

        if interaction_logging:
            if logged_hinge_values:
                flat_hinges = torch.cat(
                    [values.reshape(-1) for values in logged_hinge_values]
                )
                hinge_mean = float(flat_hinges.mean().item())
                active_fraction = float((flat_hinges > 0).float().mean().item())
            else:
                hinge_mean = 0.0
                active_fraction = 0.0
            ifa_weight = float(getattr(self.cfg, "ifa_weight", 0.1))
            self._last_ifa_interaction_stats = {
                "base_bc_loss_before_ifa": float(base_bc_loss.item()),
                "ifa_hinge_mean": hinge_mean,
                "ifa_hinge_active_fraction": active_fraction,
                "ifa_added_loss": ifa_weight * hinge_mean,
                "selected_old_tasks": tuple(logged_old_tasks),
                "current_task": int(current_task_id),
                "epoch": int(epoch),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "current_task_sample_count": int(batch_mask.sum().item()),
            }


        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )

        self.optimizer_step_and_maybe_schedule()
        return loss.item()

    def eval_observe(self, data):
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            loss = self.policy.compute_loss(data)
        return loss.item()

    def learn_one_task(self, dataset, task_id, benchmark, result_summary, save_feature=False,saving_dir=None):

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=RandomSampler(dataset),
        )

        self.start_task(task_id,train_dataloader)

        # start training

        for epoch in range(0, self.cfg.train.n_epochs + 1):

            t0 = time.time()

            if save_feature is True:
                self.policy.train()
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    self.observe(data,task_id,idx, save_feature, saving_dir)
                    max_feature_batches = getattr(
                        self.cfg, "max_feature_batches", None
                    )
                    if (
                        max_feature_batches is not None
                        and idx + 1 >= int(max_feature_batches)
                    ):
                        break
                break

            self.policy.train()
            training_loss = 0.0
            for (idx, data) in enumerate(train_dataloader):
                loss = self.observe(data,save_feature)
                training_loss += loss

            training_loss /= len(train_dataloader)

            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}"
            )
            if epoch % 3 == 0 or epoch == 49:
                if dist.get_rank() == 0:

                    model_checkpoint_name = os.path.join(self.experiment_dir, f"task{task_id}_model_{epoch}.pth")
                    
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)

            self.epoch_schedule_step(epoch)

    def reset(self):
        self.policy.reset()
