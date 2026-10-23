"""Remote stat-only remaining-allocation bound; no payload reads or server writes."""
import json
import os
import stat
import time
from pathlib import Path

Q = Path(os.environ['EM_ROOT'])
entry = json.loads((Q/'ENTRY.json').read_text())
roots = dict(entry['arms'])
roots['shared'] = str((Q/'shared').resolve())
GiB = 1024**3

def scan(root):
    found = {}
    if not root.exists():
        return found
    for base, dirs, files in os.walk(str(root), followlinks=False):
        dirs[:] = [n for n in dirs if not (Path(base)/n).is_symlink()]
        for n in files:
            p = Path(base)/n
            try:
                s = p.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(s.st_mode) and s.st_nlink == 1:
                found[str(p)] = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_blocks*512)
    return found

first = {k: scan(Path(v)) for k,v in roots.items()}
second = {k: scan(Path(v)) for k,v in roots.items()}
mounts = {'/data': dict(free=os.statvfs('/data').f_bfree*os.statvfs('/data').f_frsize, reserve=2*GiB, remaining=0),
          '/': dict(free=os.statvfs('/').f_bavail*os.statvfs('/').f_frsize, reserve=GiB, remaining=0)}
reports = {}
seen = set()
for name, path in roots.items():
    root = Path(path).resolve()
    mount = '/data' if str(root).startswith('/data/') else '/'
    stable = {p:s for p,s in second[name].items() if first[name].get(p) == s}
    credit = 0
    for s in stable.values():
        inode = s[:2]
        if inode not in seen:
            credit += s[-1]
            seen.add(inode)
    bound = int((.3 if name == 'shared' else 1.8)*GiB)
    remaining = max(0, bound-credit)
    mounts[mount]['remaining'] += remaining
    reports[name] = dict(root=str(root),mount=mount,planned_bound_bytes=bound,
                         stable_private_allocated_bytes=credit,remaining_new_bound_bytes=remaining,
                         stable_files=len(stable),unstable_or_disappeared_files=len(set(first[name])|set(second[name]))-len(stable))
for m in mounts.values():
    m['after_remaining_and_reserve'] = m['free']-m['remaining']-m['reserve']
report = dict(status='PASS_NOT_FINAL' if all(m['after_remaining_and_reserve']>=0 for m in mounts.values()) else 'CAPACITY_REVIEW_REQUIRED',
              unix=time.time(),root=str(Q),arms=reports,mounts=mounts,
              payload_reads=0,server_writes=0,forwards=0,optimizer_updates=0,environment_steps=0,
              note='Credit only stable regular private single-link files; symlinks/hardlinks excluded. Preserved recovery outside these output roots is already reflected in free space, not credited twice.')
print(json.dumps(report))
