"""Recompute the scope and outcome from the two unprivileged permission checks."""
import hashlib
import json
from pathlib import Path
import sys


def summarize(path):
    raw = path.read_bytes()
    row = json.loads(raw)
    before, after = row['before'], row['after']
    entry = 'user:' + str(row['uid'])
    expected = dict(before, **{entry: 'rw-', 'mask:': 'rw-'})
    independent = row['independent_check']
    assert before == {'user:': 'rw-', 'group:': 'rw-', 'other:': '---'}
    assert after == expected and row['changed']
    assert row['access_before'] == [False, False] and row['access_after'] == [True, True]
    assert independent['rights'] == [True, True] and independent['inode'] == row['inode']
    assert independent['account'] == row['account']
    assert f"user:{row['account']}:rw-" in independent['acl']
    return dict(verified_at=row['verified_at'], device=row['node'],
                scope='one existing device node, one named account', permission='rw-',
                account_read_write_verified=True, independent_same_inode_verified=True,
                other_acl_entries_preserved=True, evidence_sha256=hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
