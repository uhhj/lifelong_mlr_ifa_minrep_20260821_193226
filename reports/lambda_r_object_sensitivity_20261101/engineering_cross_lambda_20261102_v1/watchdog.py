import os,json,time,signal
from pathlib import Path
E=Path(__file__).resolve().parent
def proc(i):
 p=Path('/proc')/str(i['pid'])
 if not p.exists():return None
 s=(p/'stat').read_text().split(') ',1)[1].split();return dict(state=s[0],start=s[19])
g=json.loads((E/'GUARD.json').read_text());owner=json.loads((E/'STARTED.json').read_text())['identity']
(E/'WATCHDOG_READY.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time())))
while not (E/'HANDOFF_COMPLETE.json').exists():
 p=proc(owner)
 if p is None or p['start']!=owner['start'] or p['state']=='Z':
  if (E/'LIVE_COMMITTED.json').exists():
   (E/'OWNER_LOST_AFTER_LIVE_COMMIT.json').write_text(json.dumps(dict(time=time.time(),action='Keep old coordinator frozen; L02041 may already be running; audit before intervention',training_signals=0)))
  else:
   old=g['old_controller'];p=proc(old)
   if p and p['start']==old['start'] and p['state']=='T':os.kill(old['pid'],signal.SIGCONT)
   (E/'WATCHDOG_RELEASE.json').write_text(json.dumps(dict(time=time.time(),action='Release only original waiting coordinator',training_signals=0)))
  break
 time.sleep(2)
