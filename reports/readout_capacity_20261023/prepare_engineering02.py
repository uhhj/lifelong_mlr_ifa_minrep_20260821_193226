"""Preserve failed admission version; only fix argparse/output variable shadowing."""
from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent
assert json.loads((R/'snapshot_20261023_173025/LOCAL_AUDIT.json').read_text())['complete_probe_updates']==0
A=R/'engineering_attempt_01';assert not A.exists();(A/'pbd').mkdir(parents=True)
for p in [R/'deploy.py',R/'controller.py',R/'ACTIVE_ROOT.json',*sorted((R/'pbd').glob('*.py'))]:
 out=A/p.relative_to(R);out.write_bytes(p.read_bytes())
p=R/'pbd/admit.py';s=p.read_text();s=s.replace("a=old(x,pa) if k in ('G','S') else old(x);b=new512(x,pa) if k in ('G','S') else new512(x)","ref_output=old(x,pa) if k in ('G','S') else old(x);new_output=new512(x,pa) if k in ('G','S') else new512(x)").replace('assert torch.equal(a,b)\n  equivalence','assert torch.equal(ref_output,new_output)\n  equivalence');p.write_text(s)
oldroot='/data/mlr_ifa/experiment_workspaces/READOUT_CAPACITY_OBJECT41_20261023_v1';newroot=oldroot+'_engineering02'
p=R/'deploy.py';s=p.read_text().replace(oldroot,newroot)
needle="assert all(sha(p)==h for p,h in files.items())\nV2=Path"
block="""assert all(sha(p)==h for p,h in files.items())
FAILED=Path(%r)
assert sha(FAILED/'CONTRACT.json')=='69cbf420e3a4df4ac0a6a30a40ca2dc1f8069cba2598670f817f8678e9b8a026'
fc=json.loads((FAILED/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in fc['files'].items())
assert (FAILED/'controller.exit').read_text().strip()=='1' and (FAILED/'capacity_admission.exit').read_text().strip()=='1'
assert not (FAILED/'output/EVENTS.jsonl').exists() and not (FAILED/'output/private/recovery.pth').exists()
assert "AttributeError: 'Tensor' object has no attribute 'phase'" in (FAILED/'capacity_admission.log').read_text()
for pp in Path('/proc').iterdir():
 if pp.name.isdigit():
  try:ee=(pp/'environ').read_bytes().split(b'\\\\0')
  except OSError:continue
  assert ('PBD_ROOT='+str(FAILED)).encode() not in ee,'Failed version has live process'
import fcntl
with open('/data/mlr_ifa/physical_effect_probe.lock','r') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
for name in ['CONTRACT.json','controller.exit','capacity_admission.exit','capacity_admission.log','output/FAILURE.json']:files[str(FAILED/name)]=sha(FAILED/name)
V2=Path"""%oldroot
assert needle in s;s=s.replace(needle,block);p.write_text(s)
p=R/'final_scalars_readonly.py';p.write_text(p.read_text().replace(oldroot,newroot))
e=dict(status='ENGINEERING_REPAIR_PREPARED_NOT_ADMITTED',failure_snapshot='snapshot_20261023_173025',failed_root=oldroot,failed_contract='69cbf420e3a4df4ac0a6a30a40ca2dc1f8069cba2598670f817f8678e9b8a026',new_root=newroot,change='Rename two baseline-output temporaries to avoid overwriting argparse namespace; architecture, loss, optimizer, data, seed and training unchanged',formal_epochs_before_failure=0,formal_updates_before_failure=0,isolated_optimizer_calls_reached_before_report_failure=9,baseline_equivalence_forwards_reached=6,old_pass_not_claimed=True,failed_sources={str(p.relative_to(A)):hashlib.sha256(p.read_bytes()).hexdigest() for p in A.rglob('*.py')})
(R/'ENGINEERING_REPAIR_02.json').write_text(json.dumps(e,indent=2))
for p in [R/'deploy.py',R/'pbd/admit.py',R/'final_scalars_readonly.py']:ast.parse(p.read_text(),feature_version=(3,8))
tree=ast.parse((R/'deploy.py').read_text());code=next(n.left.value for n in ast.walk(tree) if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Mod) and isinstance(n.left,ast.Constant) and isinstance(n.left.value,str) and 'Q=Path(%r)' in n.left.value);ast.parse(code%('/tmp/static',{}),feature_version=(3,8))
print(json.dumps(e,indent=2))
