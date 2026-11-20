"""Install the verified release into the current Linux user's desktop."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

release=Path(sys.argv[1]).resolve(strict=True)
python=Path(sys.argv[2])
desktop=Path(subprocess.check_output(["xdg-user-dir","DESKTOP"],text=True).strip()).resolve(strict=True)
entry=desktop/"PiPER-Policy.desktop"
marker="X-PiPER-Launcher=local-base-policy"
if entry.is_symlink() or (entry.exists() and marker not in entry.read_text().splitlines()):
    raise RuntimeError("同名桌面文件已存在，请保留原文件")
content=f'''[Desktop Entry]
Version=1.0
Type=Application
Name=PiPER 策略执行台
Comment=B1、B2、B3 实时预测与现场执行
Exec="{python}" -B "{release/'desktop.py'}"
Icon=applications-science
Terminal=false
{marker}
'''
# Exercise the actual panel, without clicking device controls, in an isolated display.
smoke="""import socket,sys,os
def guard(event,args):
    if event=='socket.__new__' and args[1]==socket.AF_CAN: raise AssertionError('CAN')
    if event=='open' and isinstance(args[0],str) and args[0].startswith('/dev/video'): raise AssertionError('camera')
sys.addaudithook(guard)
sys.path.insert(0,sys.argv[1])
import tkinter as tk
from desktop import Panel,TASKS
root=tk.Tk()
panel=Panel(root)
root.update()
assert len(TASKS)==3 and panel.proc is None and panel.start.instate(['disabled'])
assert panel.preview.instate(['!disabled'])
root.destroy()
"""
import shutil
env=dict(os.environ)
env["XDG_RUNTIME_DIR"]="/run/user/"+str(os.getuid())
env["DBUS_SESSION_BUS_ADDRESS"]="unix:path="+env["XDG_RUNTIME_DIR"]+"/bus"
for line in subprocess.check_output(["systemctl","--user","show-environment"],env=env,text=True).splitlines():
    k,_,v=line.partition("=")
    if k in ("DISPLAY","XAUTHORITY","WAYLAND_DISPLAY"): env[k]=v
command=[str(python),"-B","-c",smoke,str(release)]
if shutil.which("xvfb-run"): command=["xvfb-run","-a",*command]
else: command[3]=smoke.replace("panel=Panel(root)","root.withdraw()\npanel=Panel(root)")
subprocess.run(command,check=True,timeout=20,env=env)
with tempfile.NamedTemporaryFile(dir=desktop,suffix=".desktop",delete=False,mode="w",encoding="utf8") as f:
    candidate=Path(f.name)
    f.write(content)
    f.flush()
    os.fchmod(f.fileno(),0o755)
try:
    subprocess.run(["desktop-file-validate",str(candidate)],check=True)
    os.replace(candidate,entry)
finally: candidate.unlink(missing_ok=True)
subprocess.run(["gio","set",str(entry),"metadata::trusted","true"],env=env,check=True)
info=subprocess.check_output(["gio","info","-a","metadata::trusted",str(entry)],text=True,env=env)
assert "metadata::trusted: true" in info and os.access(entry,os.X_OK) and entry.read_text()==content
print(json.dumps(dict(desktop=str(entry),release=release.name,sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),gui_smoke=True,trusted=True)))
