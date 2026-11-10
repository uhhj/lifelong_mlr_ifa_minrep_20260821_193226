"""Environment mutex around the unchanged acquisition implementation."""
from pathlib import Path
import fcntl,json,os,runpy,sys,time
Q=Path(__file__).resolve().parent
R=Path(os.environ['REVERSE_CASE_ROOT'])
sys.path.insert(0,str(Q));sys.path.insert(0,str(R))
from reverse_common import event
lockpath=Path('/data/mlr_ifa/locks/reverse_main_protocol_environment.lock');lockpath.parent.mkdir(parents=True,exist_ok=True)
with lockpath.open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    event('environment_acquired',case=R.name,purpose='training_collection',pid=os.getpid())
    ok=False
    try:runpy.run_path(str(R/'collect_native.py'),run_name='__main__');ok=True
    finally:event('environment_released',case=R.name,purpose='training_collection',pid=os.getpid(),success=ok)
