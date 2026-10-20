"""Frozen run provenance and suite arithmetic, no scientific algorithm changes."""
import json
from pathlib import Path
W=Path(__file__).resolve().parents[2]
SPEC=json.loads((W/'RUN_SPEC.json').read_text())
O=Path(SPEC['output']);P=Path(SPEC['parent']);S=Path(SPEC['parent_source'])
SEED=SPEC['seed'];SUITE=SPEC['suite'];PARM=SPEC['parent_arm'];ARM=SPEC['arm']
assert SEED in [41,42,43,44] and SUITE in ['LIBERO_OBJECT','LIBERO_GOAL']
assert PARM in ['coreset','ra_coreset'] and ARM in [PARM+'_ifa_zrv3',PARM+'_ifa_zrv3a']
COUNTS=[1562,1480,1560,1319] if SUITE=='LIBERO_OBJECT' else [997,889,904,1969]
QUOTAS=[(n+1)//2 for n in COUNTS]
LENS=[n+sum(QUOTAS[:i]) for i,n in enumerate(COUNTS)]
BATCHES=[(n+9)//10 for n in LENS]
STEPS=10*sum(BATCHES);EXPOSURES=10*sum(LENS)
assert STEPS==(10550 if SUITE=='LIBERO_OBJECT' else 7610)
assert SPEC['optimizer_steps']==STEPS and SPEC['exposures']==EXPOSURES

EPSILON=.5 if SPEC['variant']=='v3' else None
assert SPEC['variant'] in ['v3','v3a'] and SEED==41 and SUITE=='LIBERO_OBJECT'
