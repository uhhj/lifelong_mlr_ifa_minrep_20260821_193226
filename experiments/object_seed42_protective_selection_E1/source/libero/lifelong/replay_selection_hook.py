"""Explicit pre-observe integration for a separately gated E0-derived runner.

Use class E1MLR(ProtectiveSelectionMixin, MLR). The runner must configure each
stage with its verified legal catalog, and atomically checkpoint this sidecar
with the existing model/optimizer/scheduler/RNG epoch checkpoint. Merely copying
this module does not start or certify an experiment.
"""
import copy
from .replay_selection import ProtectiveReplaySelector, SelectionConfig


class ProtectiveSelectionMixin:
    def configure_e1_stage(self, task, old_catalog, current_ids, payload_loader,
                           catalog_fingerprint, selection_config=None):
        self.e1_selector = ProtectiveReplaySelector(
            self.policy, self.cfg.seed, task, old_catalog, current_ids,
            payload_loader, catalog_fingerprint, selection_config or SelectionConfig())
        self.e1_stage_step = 0
        self.e1_events = []

    def observe(self, data, current_task_id, epoch, ifa_info=None):
        selector = getattr(self, 'e1_selector', None)
        if selector is None or selector.current_task != current_task_id:
            raise RuntimeError('Stage legal catalog was not configured and verified')
        # No original sampler/loader/optimizer/scheduler calls are replaced.
        before_tasks = tuple(str(x) for x in data[1])
        refresh = selector.refresh(self.e1_stage_step)
        replaced, events = selector.replace_old_slots(data)
        assert tuple(str(x) for x in replaced[1]) == before_tasks
        before_step = self._optimizer_step_count
        loss = super().observe(replaced, current_task_id, epoch, ifa_info=ifa_info)
        assert self._optimizer_step_count == before_step + 1
        self.e1_stage_step += 1
        self.e1_events.append(dict(stage=current_task_id, epoch=epoch,
            stage_optimizer_step=self.e1_stage_step, refresh=refresh, replacements=events))
        return loss

    def pop_e1_events(self):
        events, self.e1_events = self.e1_events, []
        return events

    def e1_sidecar(self):
        return dict(stage_optimizer_step=self.e1_stage_step,
                    selector=self.e1_selector.state_dict(), pending_events=copy.deepcopy(self.e1_events))

    def restore_e1_sidecar(self, state):
        self.e1_selector.load_state_dict(state['selector'])
        self.e1_stage_step = int(state['stage_optimizer_step'])
        last = self.e1_selector.last_refresh_step
        assert last is None or 0 <= last <= self.e1_stage_step
        self.e1_events = copy.deepcopy(state['pending_events'])
