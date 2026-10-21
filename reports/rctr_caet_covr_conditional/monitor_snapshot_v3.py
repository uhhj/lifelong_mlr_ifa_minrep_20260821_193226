"""Read-only scheduler-v3 metadata; retain prior controller evidence."""
import json
from monitor_snapshot_v2 import snapshot as previous
from monitor_snapshot import Q, read
def snapshot():
    result=previous()
    result['controller_v3']=dict(pid=read(Q/'controller_v3.pid.json'),
        handoff=read(Q/'CONTROLLER_V3_HANDOFF.json'),
        exit=(Q/'controller_v3.exit').read_text().strip() if (Q/'controller_v3.exit').exists() else None)
    result['deferred']=read(Q/'CAET_GOAL_DEFERRED.json')
    return result
if __name__=='__main__':print(json.dumps(snapshot(),indent=2))
