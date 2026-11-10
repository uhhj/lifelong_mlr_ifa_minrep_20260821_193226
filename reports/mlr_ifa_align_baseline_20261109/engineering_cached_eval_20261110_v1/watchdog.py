"""Release only the waiting old owner if new owner fails before handoff."""
from runtime_cached import *
import signal
g=js(E/'GUARD.json');owner=js(E/'STARTED.json')['identity']
write(E/'WATCHDOG_READY.json',dict(pid=os.getpid(),observed=time.time()))
while not (E/'HANDOFF_COMPLETE.json').exists():
    p=proc(owner['pid'])
    if not p or p['start']!=owner['start'] or p['state']=='Z':
        old=proc(g['old_controller']['pid'])
        if old and old['start']==g['old_controller']['start'] and old['state']=='T':os.kill(old['pid'],signal.SIGCONT)
        write(E/'WATCHDOG_RELEASE.json',dict(observed=time.time(),action='Release original coordinator only; no new science launched before handoff',evaluation_signals=0));break
    time.sleep(1)
