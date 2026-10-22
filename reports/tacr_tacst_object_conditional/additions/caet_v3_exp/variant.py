import json
from pathlib import Path
ENTRY=json.loads((Path(__file__).resolve().parents[2]/'ENTRY.json').read_text())
VARIANT=ENTRY['variant'];assert VARIANT in ('TACR','TACRST')
METHOD={'TACR':'ALIGN+TAC-R','TACRST':'ALIGN+TAC-R+TAC-ST'}[VARIANT]
