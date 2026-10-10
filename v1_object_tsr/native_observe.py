"""Original loss construction; only update tail delegates to TSR."""
from libero.lifelong.algos.base import *

class NativeLossMixin:
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

        bc_loss = loss
        aux_terms = []
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
            return self.finish_native(loss, data, current_task_id, epoch, bc_loss, aux_terms)

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
                    weighted_term = ifa_weight * loss_hinge_angle.mean()
                    loss = loss + weighted_term
                    aux_terms.append(weighted_term)
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


        return self.finish_native(loss, data, current_task_id, epoch, bc_loss, aux_terms)
