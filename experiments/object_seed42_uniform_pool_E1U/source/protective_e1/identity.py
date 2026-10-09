"""Read-only identity side channel; no sampler iteration or random calls."""
from pathlib import Path
from torch.utils.data import Dataset

class IdentityDataset(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, index):
        payload, task = self.base[index]
        path = Path(self.base.file_paths[index])
        item = ('buffer/' + path.name) if path.name.startswith('t') else ('current/%s/%s' % (task, path.name))
        return payload, task, item

def identity_rows(original_ids, tasks, events):
    attempted = {e['slot']: e for e in events}
    result = []
    for slot, (item, task) in enumerate(zip(original_ids, tasks)):
        e = attempted.get(slot)
        final = e['candidate'] if e and e['applied'] else item
        result.append(dict(slot=slot, task=int(task), original_id=item, final_id=final,
            attempted=e is not None, assigned=bool(e and e['applied']),
            id_changed=final != item, ifa_veto=bool(e and e.get('reason')=='IFA_LANGUAGE_REFERENCE')))
    return result
