import subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
subprocess.run([sys.executable,'-X','utf8',str(R/'remote.py'),'snapshot_readonly.py'],cwd=R,check=True)
