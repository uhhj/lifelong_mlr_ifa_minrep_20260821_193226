"""Build external read-only audit adapters; does not execute experiment jobs."""
from pathlib import Path
import ast
R=Path(__file__).resolve().parent;P=R.parent/'20261025_ccbp_v4_full'
def put(n,s):
 if n.endswith('.py'):ast.parse(s)
 (R/n).write_text(s,encoding='utf8')
s=(P/'audit_progress.py').read_text(encoding='utf8')
s=s.replace("D=Path(sys.argv[1]);", "D=Path(sys.argv[1]);seed=int(sys.argv[2]);binding=json.loads((R/('seed%d'%seed)/'PARENT_BINDING.json').read_text(encoding='utf8'));")
start=s.index('pop={');end=s.index('\ndef outcomes',start)
s=s[:start]+"pop={int(t):x['population'] for t,x in binding['stages'].items()};valid={int(t):x['valid'] for t,x in binding['stages'].items()};old={int(t):x['old_valid'] for t,x in binding['stages'].items()}\nsplit=binding['initial_split'];counts={};rates={};admissions={};memory={};collection=[]\n"+s[end:]
s=s.replace("'DOS'","'S'").replace('41+i//10 if formal else 141','seed+i//10 if formal else seed+100').replace('+41000+', '+seed*1000+')
s=s.replace("sha(parent/('ra_coreset/dataset_manifests/task%d.json'%t))", "binding['stages'][str(t)]['manifest_sha256']")
start=s.index(" if 'INITIAL_'+arm+'.json' in L:");end=s.index("records=L.get",start)
s=s[:start]+" if 'INITIAL_'+arm+'.json' in L:\n  a=L['INITIAL_'+arm+'.json'];assert a['status']=='PASS_OFFICIAL_INITIALIZATION' and a['parent_initial_sha256']==binding['files'][binding['output']+'/ra_coreset/INITIAL_STATE.json'] and a['official_cp_sha256']==binding['official_initialization']['sha256']\n sh=L.get('SHORT_HISTORY_ADMISSION_'+arm+'.json')\n if sh:assert sh['status']=='PASS' and sh['all_state_restored'] and sh['eight_frame_legacy_exact'] and [x['length'] for x in sh['lengths']]==list(range(1,9)) and sh['extra_autograd_calls']==8 and sh['optimizer_calls']==sh['environments']==0\n"+s[end:]
put('audit_seed_progress.py',s)
s=(P/'audit_evaluation_progress.py').read_text(encoding='utf8').replace("D=Path(sys.argv[1]);","D=Path(sys.argv[1]);seed=int(sys.argv[2]);").replace('[DOS]','[S]').replace('41+group','seed+group').replace('(60-len(completed))','(20-len(completed))')
put('audit_seed_evaluation.py',s)
for p in (R/'review').glob('*.diff'):
 p.with_suffix('.txt').write_bytes(p.read_bytes())
print('External per-seed progress and evaluation adapters generated.')
