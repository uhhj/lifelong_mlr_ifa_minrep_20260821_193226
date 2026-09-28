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
        for name, param in self.policy.module.named_parameters():
            if param.requires_grad:
                print(name)

        # build scheduler
        sched_cfg = self.cfg.train.scheduler.kwargs

        if self.cfg.train.scheduler is not None:
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
        else:
            self.scheduler = None

    def map_tensor_to_device(self, data):
        """Move data to the device specified by self.cfg.device."""
        return TensorUtils.map_tensor(
            data, lambda x: safe_device(x, device=self.cfg.device)
        )

    def observe(self, data, current_task_id, epoch, ifa_info=None):
        from libero.lifelong.runbook_core import ifa_loss
        data[0] = self.map_tensor_to_device(data[0])
        self.optimizer.zero_grad()
        policy = self.policy.module if isinstance(self.policy, DDP) else self.policy
        bc, last_feat = policy.compute_loss(data[0], None, None)
        negatives = {}
        if ifa_info is not None and current_task_id in ifa_info["task_ids"]:
            tids = ifa_info["task_ids"]
            pos = tids.index(current_task_id)
            for i, tid in enumerate(tids):
                if tid != current_task_id and bool(ifa_info["selected_pair_mask"][pos, i]):
                    if tid >= current_task_id:
                        raise ValueError("Future-task IFA reference forbidden")
                    if not hasattr(self, "_ifa_reference_cache"):
                        self._ifa_reference_cache = {}
                    if tid not in self._ifa_reference_cache:
                        path = os.path.join(self.cfg.feature_dir, str(tid), "0.pt")
                        self._ifa_reference_cache[tid] = torch.load(path, map_location="cpu")["feature"][0, 0, 0, :]
                    negatives[tid] = self._ifa_reference_cache[tid]
        raw_ifa = ifa_loss(last_feat, data[0]["feature"], [int(x) for x in data[1]],
                           current_task_id, ifa_info, negatives, self.cfg.gamma)
        loss = bc + 0.1 * raw_ifa
        (self.loss_scale * loss).backward()
        grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
        lr_used = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.last_loss_log = {"bc": bc.item(), "ifa_raw": raw_ifa.item(),
                              "ifa_weighted": 0.1 * raw_ifa.item(), "total": loss.item(),
                              "grad_norm": float(grad_norm), "lr_used": lr_used,
                              "lr_next": self.optimizer.param_groups[0]["lr"]}
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

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

    def reset(self):
        self.policy.reset()
