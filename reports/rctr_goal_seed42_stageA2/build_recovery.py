"""Build a separately auditable lifecycle fix; never edit the frozen runner."""
from pathlib import Path
import difflib,hashlib,json,ast
R=Path(__file__).resolve().parent
src=(R/"runner.py").read_text(encoding="utf-8")
changes=[
 ('O=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA2_20261020")',
  'ORIGINAL=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA2_20261020")\nO=ORIGINAL/"engineering_recovery/dedicated_env_v1"'),
 (' assert not (O/"CONTRACT.json").exists(),"No blind partial restart"',
  ' assert not (O/"CONTRACT.json").exists(),"No blind partial restart"\n'
  ' assert (ORIGINAL/"runner.exit").read_text().strip()=="1"\n'
  ' old_contract=json.loads((ORIGINAL/"CONTRACT.json").read_text())\n'
  ' for path,h in old_contract["files"].items():assert sha(path)==h,path\n'
  ' expected={r["initial_index"]:r for r in json.loads((ORIGINAL/"SNAPSHOT_INVENTORY.json").read_text())}'),
 (' env=None;snapshots={};gates=[];inventory=[]',' env=None;environments={};snapshots={};gates=[];inventory=[]'),
 ('  ledger("environment_construct_start")\n'
  '  env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=c.data.img_h,camera_widths=c.data.img_w)\n'
  '  ledger("environment_construct_complete")\n'
  '  for index in (20,21,22):',
  '  for index in (20,21,22):\n'
  '   ledger("environment_construct_start",initial_index=index)\n'
  '   env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=c.data.img_h,camera_widths=c.data.img_w)\n'
  '   environments[index]=env\n'
  '   ledger("environment_construct_complete",initial_index=index)'),
 ('   inventory.append(row)',
  '   for key in ("physical_sha256","observation_hashes","prefix_noise_sha256","initial_sha256","eligible"):\n'
  '    assert row[key]==expected[index][key],("recollected_identity",index,key)\n'
  '   row["recollected_identity"]="EXACT_MATCH_ORIGINAL"\n'
  '   inventory.append(row)'),
 ('    gates.append(gate(env,policy,snapshots[index],index,instr,mask,c))',
  '    # Gate all snapshots only after every environment has been constructed.'),
 ('  assert snapshots,"No eligible fixed snapshot; no replacement sampling allowed"',
  '  assert snapshots,"No eligible fixed snapshot; no replacement sampling allowed"\n'
  '  for index,snap in snapshots.items():\n'
  '   gates.append(gate(environments[index],policy,snap,index,instr,mask,c))'),
 ('  for index,snap in snapshots.items():\n   for rep in range(3):',
  '  for index,snap in snapshots.items():\n   env=environments[index]\n   for rep in range(3):'),
 ('  if env is not None:env.close();ledger("environment_closed")',
  '  for index,env in environments.items():\n   env.close();ledger("environment_closed",initial_index=index)')
]
patched=src
for a,b in changes:
 assert patched.count(a)==1,a
 patched=patched.replace(a,b)
ast.parse(patched)
(R/"recovery_runner.py").write_text(patched,encoding="utf-8",newline="\n")
diff="".join(difflib.unified_diff(src.splitlines(True),patched.splitlines(True),fromfile="frozen/runner.py",tofile="external/recovery_runner.py"))
(R/"RECOVERY_DIFF.patch").write_text(diff,encoding="utf-8")
(R/"RECOVERY_SOURCE.json").write_text(json.dumps(dict(
 original_sha256=hashlib.sha256(src.encode()).hexdigest(),
 recovery_sha256=hashlib.sha256(patched.encode()).hexdigest(),
 replacements=len(changes),status="SOURCE_BUILD_RECORD_NOT_RUNTIME_STATUS"),indent=2)+"\n",encoding="utf-8")
print(diff)
