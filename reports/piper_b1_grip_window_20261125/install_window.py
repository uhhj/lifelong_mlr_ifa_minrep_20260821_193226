"""Verify staged or installed sources; update the existing station under its session lock."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

FILES = ("grip_window.py", "bus.py", "agilex_arm.py", "runner.py")
ORIGINAL = {
    "bus.py": "c78b4d6f9ee5b9e3c139af8659d2f956755e4c4c6fb5837588781ee898e32428",
    "agilex_arm.py": "6a57c398a4cc77af64f93bb1d141f393307e6a95aa997f185ec1b2a3da5ae059",
    "runner.py": "85b30189cf9dcd5f81f83fc29b5de162f466673334d2070a4549a6e709bcaea3",
}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def put(path, data, mode):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        temp = Path(out.name)
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    try:
        temp.chmod(mode)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def main(work, stage, action):
    station = work / "piper_test_console_sdk_v1"
    candidate = stage / "candidate"
    manifest = json.loads((stage / "manifest.json").read_bytes())
    assert set(manifest) == set(FILES)
    assert manifest == {n:digest(candidate/n) for n in FILES}
    python = (station/"sdk_python").read_text().strip()
    def verify(source):
        data = subprocess.check_output([python, "-B", str(stage/"verify_window.py"), str(source),
                                        str(station), str(work/"policy_runs")], timeout=90, text=True)
        return json.loads(data.splitlines()[-1])
    test = verify(candidate)
    if action == "test":
        return test
    if action != "install":
        raise ValueError(action)
    lock_path = work.parent / "synchronized_capture/recorder.lock"
    backup = stage/"before"
    backup.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = {n:digest(station/n) for n in FILES if (station/n).exists()}
        if current != manifest:
            # Recover an interrupted install only if all pre-existing sources have
            # verified originals and every live file matches an old or staged hash.
            if current != ORIGINAL:
                if not all((backup/n).is_file() and digest(backup/n) == h for n,h in ORIGINAL.items()):
                    raise ValueError("live station differs from the reviewed version")
                if set(current) != set(FILES) or any(v not in (manifest[n],ORIGINAL.get(n)) for n,v in current.items()):
                    raise ValueError("station changed independently")
            for n,h in ORIGINAL.items():
                if not (backup/n).exists():
                    assert digest(station/n) == h
                    put(backup/n,(station/n).read_bytes(),0o600)
                assert digest(backup/n) == h
            try:
                for n in FILES:
                    dest = station/n
                    mode = dest.stat().st_mode & 0o777 if dest.exists() else 0o644
                    put(dest,(candidate/n).read_bytes(),mode)
                assert {n:digest(station/n) for n in FILES} == manifest
                installed = verify(station)
                assert installed == test
            except BaseException:
                # Old runner first. Auxiliary module is inert and can remain on disk.
                for n in reversed(FILES[1:]):
                    put(station/n,(backup/n).read_bytes(),(station/n).stat().st_mode & 0o777)
                raise
        else:
            assert verify(station) == test
        return dict(verification=test, installed=manifest, changed_profile_only=test["profile"],
                    desktop_launcher="PiPER-Policy.desktop", test_mode="no sockets or actuators")


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]),ensure_ascii=False))
