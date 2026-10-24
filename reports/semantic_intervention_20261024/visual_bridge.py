"""Read-only authenticated in-memory contact sheets; never save private imagery locally."""
import getpass,json,hashlib,datetime,threading
from pathlib import Path
from http.server import HTTPServer,BaseHTTPRequestHandler
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/RA_ALIGN_PAIRED_GOAL_20261024_v1_engineering02'
C=paramiko.SSHClient();C.load_system_host_keys();C.set_missing_host_key_policy(paramiko.RejectPolicy())
pw=getpass.getpass('SSH password (memory only): ')
C.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
def remote(code):
 i,o,e=C.exec_command('/data/mlr_ifa/env/bin/python -B -',timeout=120);i.write(code);i.flush();i.channel.shutdown_write();raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
 if rc:raise RuntimeError(err)
 return raw
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_POST(self):
  try:
   req=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   kind=req.get('kind','sheet');assert kind in ['meta','sheet','close']
   if kind=='close':self.send_response(200);self.end_headers();self.wfile.write(b'closed');threading.Thread(target=self.server.shutdown).start();return
   code=(R/'visual_readonly.py').read_text(encoding='utf8')
   raw=remote('REQUEST='+repr(req)+'\n'+code)
   if kind=='meta':(R/'ASSET_PREFLIGHT.json').write_bytes(raw)
   else:
    log={'time':datetime.datetime.now().astimezone().isoformat(),'request':req,'script_sha256':hashlib.sha256(code.encode()).hexdigest(),'png_sha256':hashlib.sha256(raw).hexdigest(),'png_bytes':len(raw)}
    with (R/'VIEW_LOG.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(log)+'\n')
   self.send_response(200);self.send_header('Content-Type','application/json' if kind=='meta' else 'image/png');self.end_headers();self.wfile.write(raw)
  except Exception as e:
   self.send_response(500);self.end_headers();self.wfile.write(str(e).encode())
try:
 server=HTTPServer(('127.0.0.1',0),Handler);print('READY_PORT='+str(server.server_port),flush=True);server.serve_forever()
finally:C.close()
