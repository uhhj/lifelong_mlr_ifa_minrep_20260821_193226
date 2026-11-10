"""Keep original metric recomputation; explicitly carry the cache boundary."""
from pathlib import Path
import json,sys,subprocess,hashlib
R=Path(__file__).resolve().parent;B=R.parent;D=Path(sys.argv[1])
J=lambda p:json.loads(p.read_text(encoding='utf8'))
a=J(D/'LOCAL_FINAL_AUDIT.json');assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
boundary=a['evaluation_cache_boundary'];assert not boundary['bitwise_equivalence_claim']
subprocess.run([sys.executable,str(B/'compare_baseline.py'),str(D)],check=True)
out=J(D/'BASELINE_COMPARISON.json');out['evaluation_cache_boundary']=boundary
note='Future unstarted evaluations used validated single-frame cache at atol=rtol=1e-4; completed legacy evaluations were retained. Original global Normal sampling and done behavior were preserved. This numerical execution boundary is not a claim of identical closed-loop outcomes.'
out['limits'].append(note)
out['cache_local_final_audit_sha256']=hashlib.sha256((D/'LOCAL_FINAL_AUDIT.json').read_bytes()).hexdigest()
(D/'BASELINE_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
with (D/'BASELINE_COMPARISON.md').open('a',encoding='utf8') as f:
 f.write('\n## Recorded evaluation execution boundary\n\n'+note+'\n\nLegacy full-window cases: '+', '.join(boundary['legacy_full_window_cases'])+'\n\nCached cases: '+', '.join(boundary['cached_cases'])+'\n')
print('PASS_COMPARISON_WITH_EXPLICIT_CACHE_BOUNDARY')
