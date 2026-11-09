"""Serialize the unchanged accepted evaluator, retaining argv and cwd."""
from runtime import *
import fcntl,sys
case=sys.argv[1];assert case in CASES
lock=(Q/'environment.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
event('environment_acquired',case=case,pid=os.getpid())
code=None
try:
 code=subprocess.call([PY,'-B','-m','coreset_exp.evaluate','random'],cwd=Q/'cases'/case/'source',env=env(case))
finally:
 event('environment_released',case=case,pid=os.getpid(),code=code);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
sys.exit(code)
