"""Local one-time adaptation of already validated ZRv2; never runs an experiment."""
from pathlib import Path
import json,hashlib,difflib,shutil,ast
N=Path(__file__).resolve().parent;R=N.parent;OLD=R.parent/'20261018_zrv2_seed43';T=N/'zr_template'
def replace(s,a,b):
 assert a in s,repr(a);return s.replace(a,b)
def main():
 assert not T.exists();(T/'source/zr2_exp').mkdir(parents=True);(T/'attachment').mkdir()
 for p in (OLD/'attachment/zr_v2_seed43_plan').iterdir():
  if p.suffix in ['.py','.md','.yaml']:shutil.copy2(p,T/'attachment'/p.name)
 changes={};diff=[]
 for p in (OLD/'source/zr2_exp').glob('*.py'):
  original=p.read_text();s=original
  if p.name=='common.py':
   start=s.index("W=Path(");end=s.index('sys.dont_write_bytecode')
   s=s[:start]+"from .params import *\nos.environ.update(CORESET_OUT=str(P),CORESET_SEED=str(SEED),PYTHONDONTWRITEBYTECODE='1')\n"+s[end:]
   s=replace(s,'control_seed(43)','control_seed(SEED)')
   s=replace(s,"cfg('coreset')","cfg(PARM)")
   s=replace(s,'MeanCalibrationBank(range(6,t+1),w)','MeanCalibrationBank(range(6,t+1),w,experiment_seed=SEED)')
  for a,b in [("P/'coreset/storage'","P/PARM/'storage'"),("P/'coreset/buffer'","P/PARM/'buffer'"),("P/'coreset'","P/PARM"),("P/'coreset/evaluation'","P/PARM/'evaluation'")]:
   s=s.replace(a,b)
  if p.name=='calibrate.py':
   s=replace(s,"str(O/'buffer'),43,start=6","str(O/'buffer'),SEED,start=6")
   s=replace(s,'[1562,2261,3081,3620][t-6]','LENS[t-6]');s=replace(s,'[157,227,309,362][t-6]','BATCHES[t-6]')
   s=replace(s,'globalstep==10550 and exposures==105240','globalstep==STEPS and exposures==EXPOSURES')
  if p.name=='integration.py':s=replace(s,'optimizer(bank,2270)','optimizer(bank,10*BATCHES[1])')
  if p.name=='evaluate.py':
   s=replace(s,'control_seed(43)','control_seed(SEED)');s=replace(s,'see=43','see=SEED');s=replace(s,"arm='coreset_ifa_zrv2'","arm=ARM")
  if p.name=='audit.py':
   s=replace(s,'from compare_metrics import compare','from branch_metrics import compare')
   s=s.replace('10550','STEPS').replace('105240','EXPOSURES').replace('[1562,2261,3081,3620][t-6]','LENS[t-6]')
   s=s.replace("'Three-seed/five-seed expansion is conditional on all metrics and process acceptance.'","'Whole-three-seed gate: mean AUC delta positive and at least two positive seed deltas; engineering all PASS.'")
   s=s.replace("'Historical RTX4090 parent vs current RTX3090; global action RNG not isolated.'","'Historical hardware may differ; global action RNG not isolated; see parent binding.'")
  if p.name=='freeze.py':
   s=s.replace("name='ZR_v2_OBJECT_seed43_repeat1_parent',seed=43","name=ARM+'_'+SUITE+'_seed'+str(SEED),seed=SEED")
   s=s.replace('optimizer_steps=10550,sample_exposures=105240','optimizer_steps=STEPS,sample_exposures=EXPOSURES')
   s=s.replace("conditional_extension='all four exact metric criteria and all engineering acceptance; user 2026-10-18 authorization'","conditional_extension='User 2026-10-19 whole-three-seed mean AUC positive and at least 2/3 positive; all engineering PASS'")
   s=s.replace("cfg('coreset')","cfg(PARM)").replace("spec=__import__('yaml').safe_load((W/'attachment/spec.yaml').read_text())","spec=SPEC").replace('actual=dict(seed=43','actual=dict(seed=SEED')
   s=replace(s,"W/'pipeline.py',W/'env.sh'","W/'pipeline.py',W/'env.sh',W/'RUN_SPEC.json',Path(SPEC['controller_contract'])")
  ast.parse(s);(T/'source/zr2_exp'/p.name).write_text(s,encoding='utf-8')
  changes[p.name]=dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),new_sha256=hashlib.sha256(s.encode()).hexdigest(),changed=s!=original)
  diff.extend(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='original/'+p.name,tofile='adapted/'+p.name))
 # Admission uses the same real language/gate and CUDA checks, but dynamically binds exact parent records.
 s=(OLD/'admission.py').read_text();start=s.index('W=Path(');end=s.index('sys.dont_write_bytecode')
 s=s[:start]+"from zr2_exp.params import *\nos.environ.update(CORESET_OUT=str(P),CORESET_SEED=str(SEED),PYTHONDONTWRITEBYTECODE='1')\n"+s[end:]
 s=s.replace('from compare_metrics import load_records,metrics,LOCKED','from branch_metrics import load_records,metrics')
 s=s.replace("P/'coreset/evaluation'","P/PARM/'evaluation'").replace('assert metrics(rates)==LOCKED',"assert SPEC['parent_engineering_file'] and json.loads(Path(SPEC['parent_engineering_file']).read_text())['status']=='PASS'")
 s=s.replace("assert binding['modal_sha256']=='c379b9930387abc9eed54e8a9f83394f2f8be80ee1536921a53ce2cab66ea78d'","assert cfg(PARM).benchmark_name==SUITE and cfg(PARM).seed==SEED\n        assert all(cfg(PARM).policy[k] is True for k in ('use_vision','use_language','use_state'))\n        assert cfg(PARM).gamma==(.3 if SUITE=='LIBERO_OBJECT' else .7)")
 s=s.replace("'resolved_coreset.json'","'resolved_'+PARM+'.json'")
 s=s.replace("P/'coreset/models'","P/PARM/'models'").replace("P/'coreset/storage'","P/PARM/'storage'").replace("P/'coreset/buffer'","P/PARM/'buffer'")
 s=s.replace('[781,740,780,660][t-6]','QUOTAS[t-6]')
 s=s.replace("inputs.update(c['inputs'])","inputs.update(c.get('inputs',{}));inputs.update(c.get('files',{}));inputs[SPEC['parent_engineering_file']]=sha(SPEC['parent_engineering_file'])")
 s=s.replace('control_seed(43)','control_seed(SEED)')
 s=s.replace("binding.update(parent=str(P),source=str(S)","binding.update(parent=str(P),parent_arm=PARM,seed=SEED,suite=SUITE,source=str(S)")
 s=s.replace("fm=json.loads((P/'FEATURES.json').read_text());","fm=json.loads((P/'FEATURES.json').read_text());assert [fm['counts'][str(t)] for t in range(6,10)]==COUNTS;")
 ast.parse(s);(T/'admission.py').write_text(s,encoding='utf-8')
 # Pipeline body is unchanged, only output root is supplied by the frozen spec.
 s=(OLD/'pipeline.py').read_text().replace("O=Path('/root/mlr_ifa/experiments/ZRv2_OBJECT_seed43_20261018')","O=Path(json.loads((W/'RUN_SPEC.json').read_text())['output'])")
 (T/'pipeline.py').write_text(s,encoding='utf-8')
 (N/'ZR_ADAPTATION_DIFF.patch').write_text(''.join(diff),encoding='utf-8');(N/'ZR_ADAPTATION_HASHES.json').write_text(json.dumps(changes,indent=2)+'\n')
 print(json.dumps(changes,indent=2))
if __name__=='__main__':main()
