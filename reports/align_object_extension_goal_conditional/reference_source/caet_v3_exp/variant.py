import json
from pathlib import Path
ENTRY=json.loads((Path(__file__).resolve().parents[2]/'ENTRY.json').read_text())
VARIANT=ENTRY['variant'];assert VARIANT in ('TAC','EFTR','ALIGN')
METHOD={'TAC':'CAET-v3-TAC','EFTR':'RA-EFTR-v3','ALIGN':'RA-TAC-Align'}[VARIANT]
