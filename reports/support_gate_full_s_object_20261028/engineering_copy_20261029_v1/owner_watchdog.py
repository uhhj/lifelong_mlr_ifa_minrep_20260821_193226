"""Fail-open queue guard before committed handoff; never signal child training."""
import os,signal,json,time
from pathlib import Path
E=Path(__file__).resolve().parent;g=json.loads((E/'GUARD.json').read_text());owner=json.loads((E/'STARTED.json').read_text())
def stat(pid):
 try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
 except OSError:return None
(E/'OWNER_WATCHDOG_READY.json').write_text(json.dumps(dict(time=time.time(),pid=os.getpid())))
while not (E/'HANDOFF_COMMITTED.json').exists():
 s=stat(owner['pid'])
 if s is None or s[0]=='Z' or s[19]!=owner['identity']['start']:
  old=stat(g['controller']['pid'])
  if old and old[0]=='T' and old[19]==g['controller']['start']:
   os.kill(g['controller']['pid'],signal.SIGCONT)
   (E/'OWNER_FAILURE_RELEASE.json').write_text(json.dumps(dict(time=time.time(),action='Released original coordinator only; no optimized seed launch')))
  break
 time.sleep(5)
