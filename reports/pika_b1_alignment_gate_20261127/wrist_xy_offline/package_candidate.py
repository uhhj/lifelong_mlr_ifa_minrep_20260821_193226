"""Bind the changed files to a copy of the existing stop-at-deadline profile."""
from pathlib import Path
import hashlib
import json


def package(root):
    def digest(name):
        return hashlib.sha256((root / name).read_bytes()).hexdigest()
    profile_path = root / 'profiles.json'
    profiles = json.loads(profile_path.read_bytes())
    profile = profiles['B1-ode50-aligned-20261125']
    assert profile['end_mode'] == 'stop' and profile['duration'] == 6.304570837
    old = profile['sha256']
    config = json.loads((root / 'camera_models.json').read_bytes())['train']['calibration']
    (root / 'wrist_camera.json').write_bytes(json.dumps({k:config[k] for k in ('R','f')}).encode()+b'\n')
    bundle = profile['alignment']
    for n in ('wrist_camera.json','wrist_cue.py'):
        bundle['station_files'][n] = digest(n)
    bundle['backend_sha256'] = digest('ode_alignment.py')
    new = hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(',',':'), ensure_ascii=True).encode()).hexdigest()
    grip = root / 'grip_window.py'
    contents = grip.read_bytes()
    assert contents.count(old.encode()) == 1
    grip.write_bytes(contents.replace(old.encode(), new.encode()))
    profile['sha256'] = new
    profile_path.write_bytes(json.dumps(profiles,ensure_ascii=False,indent=1).encode()+b'\n')
    print('candidate',new,flush=True)


if __name__ == '__main__':
    package(Path(__file__).resolve().parent)
