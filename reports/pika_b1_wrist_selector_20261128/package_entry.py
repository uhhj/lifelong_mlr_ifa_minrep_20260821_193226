"""Build the new selector entry from the frozen candidate and its actual replay evidence."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

OLD='B1-ode50-aligned-20261125'
NEW='B1-ode50-wrist-20261127'
VERSION='11-27 腕部纠偏'
def make(live, frozen, audit, destination):
    catalog=json.loads((live/'profiles.json').read_bytes())
    assert NEW not in catalog
    cfg=copy.deepcopy(json.loads((frozen/'profiles.json').read_bytes())[OLD])
    result=json.loads(audit.read_bytes())
    assert (cfg['sha256']==result['sha256']=='27fca3b2b3252827076d08aff0a122e3ec12004e07959ce7992c0b74e0b3d88d'
            and catalog[OLD]['sha256']==result['active_station']['sha256'] and cfg['end_mode']=='stop')
    assert not result['expert_close_differences']
    assert not any(p['task']=='B1' and p['family']=='ODE' and p['version']==VERSION for p in catalog.values())
    cfg['version']=VERSION
    catalog[NEW]=cfg
    (destination/'profiles.json').write_bytes((json.dumps(catalog,ensure_ascii=False,indent=1)+'\n').encode())
    manifest=cfg['alignment']
    files={'ode_alignment.py':('ode_wrist_alignment.py',manifest['backend_sha256'])}
    for n in ('wrist_cue.py','wrist_camera.json'):
        files[n]=(n,manifest['station_files'][n])
    for n,(target,expected) in files.items():
        assert hashlib.sha256((frozen/n).read_bytes()).hexdigest()==expected
        shutil.copyfile(frozen/n,destination/target)
    episodes=result['controls']['episodes']['other']
    assert len(episodes)==33 and sum(x['close_accepted'] for x in episodes)==19
    receipt=dict(profile=NEW,sha256=cfg['sha256'],replay_evidence_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
                 windows=[{'episode':x['episode'],'close_accepted':x['close_accepted'],'close_tick':x['close_tick']} for x in episodes])
    (destination/f'verified-{NEW}.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())

if __name__=='__main__':
    import sys
    make(*(Path(a) for a in sys.argv[1:]))
