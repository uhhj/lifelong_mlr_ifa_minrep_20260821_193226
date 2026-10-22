import subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(R/'remote.py'),'snapshot_readonly.py'],check=True)
