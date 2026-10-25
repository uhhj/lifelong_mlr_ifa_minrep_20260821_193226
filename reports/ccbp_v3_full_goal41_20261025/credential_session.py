"""Local interactive credential memory only; executes explicitly queued reviewed scripts.

No listener, password file, command-line secret, or automatic experiment dispatch.
The password dialog masks input. Queue files contain only script names/hashes.
"""
import contextlib, datetime, hashlib, io, json, os, time, sys, getpass
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog
import remote
R=Path(__file__).resolve().parent;D=R/'.session_control'
ALLOWED={'preflight_readonly.py','check_readonly.py','capacity_readonly.py','deploy_remote.py','final_readonly.py'}

def write(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2),encoding='utf8');os.replace(q,p)

def main():
    D.mkdir(exist_ok=True)
    write(D/'status.json',dict(state='WAITING_FOR_MASKED_INPUT',pid=os.getpid()))
    if '--console' in sys.argv:
        secret=getpass.getpass('SSH credential (no echo, memory only): ')
    else:
        app=tk.Tk();app.withdraw();app.attributes('-topmost',True)
        secret=simpledialog.askstring('CCBP SSH — memory only','SSH root@jq1.9gpu.com:11920\n密码仅存于本进程内存，不写入文件或日志。',show='*',parent=app)
        app.destroy()
    if not secret:
        write(D/'status.json',dict(state='INPUT_CANCELLED',pid=os.getpid()));return
    remote.getpass.getpass=lambda prompt:secret
    write(D/'status.json',dict(state='READY_IN_MEMORY',pid=os.getpid()))
    deadline=time.monotonic()+48*3600
    try:
        while time.monotonic()<deadline and not (D/'STOP').exists():
            for p in sorted(D.glob('request_*.json')):
                result=p.with_name('result_'+p.name[len('request_'):])
                if result.exists():continue
                request=json.loads(p.read_text(encoding='utf8'));name=request['script']
                assert name in ALLOWED and Path(name).name==name
                assert hashlib.sha256((R/name).read_bytes()).hexdigest()==request['sha256']
                write(D/'status.json',dict(state='RUNNING_REQUEST',pid=os.getpid(),request=p.name,script=name))
                buffer=io.StringIO();ok=True;error=None
                try:
                    with contextlib.redirect_stdout(buffer),contextlib.redirect_stderr(buffer):remote.run(name)
                except Exception as e:ok=False;error=dict(type=type(e).__name__,message=str(e))
                write(result,dict(ok=ok,error=error,output=buffer.getvalue()[-12000:],completed=datetime.datetime.now().astimezone().isoformat()))
                write(D/'status.json',dict(state='READY_IN_MEMORY',pid=os.getpid(),completed_request=p.name))
            time.sleep(1)
    finally:
        secret=None;remote.getpass.getpass=lambda prompt:None
        write(D/'status.json',dict(state='CLOSED',pid=os.getpid()))

if __name__=='__main__':main()
