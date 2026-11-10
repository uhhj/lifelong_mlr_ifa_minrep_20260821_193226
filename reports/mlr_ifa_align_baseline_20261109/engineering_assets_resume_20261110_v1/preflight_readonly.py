from pathlib import Path
import sys,json,hashlib,time,re,xml.etree.ElementTree as ET
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2');P=Path('/data/mlr_ifa/experiments')/Q.name
sys.path.insert(0,str(Q))
from runtime import verify,sha,processes,CASES,capacity
verify();assert not processes([Q,P])
assert (Q/'controller.exit').read_text().strip()=='1'
assert sha(Q/'FAILURE.json')=='b74b059b2c377de0091988bb77db57dbfa0bb710586d55ddce682cde2a024b0e'
A=Path('/data/mlr_ifa/src/libero/libero/assets');assert A.is_dir()
manifest={str(p.relative_to(A)):sha(p) for p in sorted(A.rglob('*')) if p.is_file()}
assert 'scenes/libero_floor_base_style.xml' in manifest
assert not any(Path(n).suffix in ['.py','.yaml','.sh'] for n in manifest),'Asset link would change frozen source-file set'
for p in A.rglob('*.xml'):ET.parse(p)
parents={};missing=[]
for case in CASES:
 inp=json.loads((P/case/'INPUT.json').read_text());pa=Path(inp['parent'])/'source/libero/libero/assets'
 parents[case]=dict(path=str(pa),exists=pa.exists(),resolved=str(pa.resolve()))
 if pa.exists():
  pm={str(p.relative_to(pa)):sha(p) for p in sorted(pa.rglob('*')) if p.is_file()};assert pm==manifest,case
 target=Q/'cases'/case/'source/libero/libero/assets';assert not target.exists() and not target.is_symlink();missing.append(str(target))
log=(P/'OBJECT_41/evaluate.log').read_text()
errors=re.findall(r'FileNotFoundError:.*',log)
assert len(errors)==10 and all('libero_floor_base_style.xml' in x for x in errors),(len(errors),errors)
attempts=[json.loads(x) for x in (P/'OBJECT_41/random/evaluation/attempts_stage6_task6.jsonl').read_text().splitlines()]
assert [x['event'] for x in attempts]==['environment_construct_start','environment_construct_complete','vector_attempt_start','technical_exception']
assert not list((P/'OBJECT_41/random/evaluation').glob('trials*'))
files={str(p.relative_to(P/'OBJECT_41/random/evaluation')):sha(p) for p in (P/'OBJECT_41/random/evaluation').rglob('*') if p.is_file()}
assert 'rng_before_stage6_task6.pth' in files
print(json.dumps(dict(status='PASS_READONLY_ASSETS_AND_PRE_ROLLOUT_FAILURE',observed=time.time(),asset_root=str(A),asset_manifest=manifest,asset_bytes=sum(p.stat().st_size for p in A.rglob('*') if p.is_file()),parent_assets=parents,missing_links=missing,worker_file_not_found_count=len(errors),attempt_events=[x['event'] for x in attempts],failed_evaluation_hashes=files,log_sha256=sha(P/'OBJECT_41/evaluate.log'),capacity=capacity(),new_optimizer=0,new_environment=0,note='All10workers failed XML file open in _load_model before simulator construction. Parent vector_attempt_start exists, but no policy-action rollout completed; retain attempt and saved pre-cell RNG.')))
