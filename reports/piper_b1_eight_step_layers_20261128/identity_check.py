from common import *
live=work/'piper_test_console_pika_b1_full_v1'
field=read(old/'profiles.json')['B1-ode50-aligned-20261125']
cand=read(frozen/'candidate.json')
active=read(live/'profiles.json')['B1-ode50-wrist-20261127']
assert field['sha256']==active['sha256']=='27fca3b2b3252827076d08aff0a122e3ec12004e07959ce7992c0b74e0b3d88d'
assert field['alignment']['base_sha256']==cand['alignment']['base_sha256']=='2e14ef23b141c697f0ffc201bea42650b02569601d7fadf6f08b5a1204c7aa20'
dump('identity_check.json',dict(field=field['sha256'],candidate=cand['sha256'],ode=field['alignment']['base_sha256'],
                                files={n:rp.sha(live/n) for n in ('runner.py','bus.py','b1_placement.py','ode_wrist_alignment.py')},
                                candidate_backend=rp.sha(frozen/'ode_tracked.py')))
