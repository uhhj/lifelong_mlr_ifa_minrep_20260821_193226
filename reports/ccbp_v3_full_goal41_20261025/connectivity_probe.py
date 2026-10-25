"""One bounded unauthenticated SSH banner probe. No credentials/remote commands."""
import datetime, hashlib, json, socket, time
from pathlib import Path
R=Path(__file__).resolve().parent
result=dict(observation=datetime.datetime.now().astimezone().isoformat(),host='jq1.9gpu.com',port=11920,
            authenticated=False,remote_commands=0,training_updates=0,environment_calls=0)
start=time.monotonic()
try:
    with socket.create_connection(('jq1.9gpu.com',11920),timeout=15) as s:
        result['tcp_connected']=True;s.settimeout(15)
        banner=s.recv(256)
        result.update(banner_bytes=len(banner),ssh_banner=bool(banner.startswith(b'SSH-')),
                      banner_sha256=hashlib.sha256(banner).hexdigest())
except Exception as e:result.update(error_type=type(e).__name__,error=str(e))
result['seconds']=time.monotonic()-start
result['status']='SSH_BANNER_AVAILABLE_AUTH_NOT_TESTED' if result.get('ssh_banner') else 'SSH_UNAVAILABLE_BEFORE_AUTH'
out=R/('connectivity_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
(out/'EVIDENCE.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(out);print(json.dumps(result,indent=2))
