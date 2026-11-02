import json,os,time,signal
from pathlib import Path
E=Path(__file__).resolve().parent
js=lambda p:json.loads(p.read_text())
def proc(i):
 p=Path('/proc')/str(i['pid'])
 if not p.exists():return None
 s=(p/'stat').read_text().split(') ',1)[1].split();return dict(state=s[0],start=s[19])
g=js(E/'GUARD.json');owner=js(E/'STARTED.json')['identity'];(E/'WATCHDOG_READY.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time())))
while True:
 if (E/'HANDOFF_COMPLETE.json').exists():break
 p=proc(owner)
 if p is None or p['start']!=owner['start'] or p['state']=='Z':
  if (E/'LIVE_COMMITTED.json').exists():
   (E/'OWNER_LOST_AFTER_LIVE_COMMIT.json').write_text(json.dumps(dict(time=time.time(),action='Keep old coordinators frozen; never duplicate seed43; no training signals',training_signals=0)))
  else:
   flag=E/'HOOK_ENABLED.json'
   if flag.exists():assert flag.read_text()=='{}';flag.unlink()
   for key in ['parallel','watchdog']:
    old=g['controllers'][key];p=proc(old)
    if p and p['start']==old['start'] and p['state']=='T':os.kill(old['pid'],signal.SIGCONT)
   (E/'WATCHDOG_RELEASE.json').write_text(json.dumps(dict(time=time.time(),action='Restore prior waiting coordinator/watchdog only; disable new startup lock',training_signals=0)))
  break
 time.sleep(2)
