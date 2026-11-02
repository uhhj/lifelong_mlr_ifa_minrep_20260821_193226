"""UNDEPLOYED: release only waiting original coordinator if owner fails pre-handoff."""
import json,os,time,signal
from pathlib import Path
E=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text())
def proc(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 s=(p/'stat').read_text().split(') ',1)[1].split();return dict(state=s[0],start=s[19])
g=js(E/'GUARD.json');owner=js(E/'STARTED.json')['identity']
(E/'WATCHDOG_READY.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time())))
while True:
 if (E/'HANDOFF_COMPLETE.json').exists():break
 old=proc(g['controller']['pid'])
 if old is None or old['start']!=g['controller']['start'] or old['state']=='Z':break
 current=proc(owner['pid'])
 if current is None or current['start']!=owner['start'] or current['state']=='Z' or (E/'FAILURE.json').exists():
  if old['state']=='T':
   os.kill(g['controller']['pid'],signal.SIGCONT)
   (E/'WATCHDOG_RELEASE.json').write_text(json.dumps(dict(time=time.time(),action='SIGCONT original waiting coordinator only',training_signals=0)))
  break
 time.sleep(2)
