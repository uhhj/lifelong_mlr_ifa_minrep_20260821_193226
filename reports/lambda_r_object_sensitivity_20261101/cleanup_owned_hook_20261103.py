BINDING={'acceptance_status': 'PASS_COMPLETE_LOCAL_ACCEPTANCE', 'acceptance_sha256': '46134f9341193cb349f0ed3e9f176e7da204a862314e4adfec0c5dfdf131c7e7', 'local_final_audit_sha256': 'da0e32a1ec0f7830f05d870b3c84af77c09c46cd23b3fe36288313734c89d86a', 'report_sha256': '64cf0ee9174ae0753d4ee785ab4cf4df21e2089d28ac23be91e83aab196ebce5', 'publication_status': 'PUSH_VERIFIED', 'publication_commit': 'f63efeb3b6a11857fb40a3747e0894dd7596a6a0', 'publication_remote': 'f63efeb3b6a11857fb40a3747e0894dd7596a6a0'}
"""Authorized post-acceptance cleanup of exactly two owned OBJECT hook files."""
import hashlib,json,os,time
from pathlib import Path
# The concrete dispatcher prepends a reviewed local acceptance/publication binding.
assert BINDING['acceptance_status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
assert BINDING['publication_status']=='PUSH_VERIFIED' and BINDING['publication_commit']==BINDING['publication_remote']
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
E=Q/'engineering_live_parallel_3090_20261102_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (Q/'engineering_cross_lambda_20261102_v1/controller.exit').read_text().strip()=='0'
assert not (Q/'OWNED_HOOK_CLEANUP.json').exists(),'Already cleaned: retrieve existing receipt; do not repeat'
assert sha(E/'CONTRACT.json')=='ff6eb2a0aee3380dfec5ee1c5473d11109f57a867d28b2d3494c9f31fd733193'
c=json.loads((E/'CONTRACT.json').read_text())
assert all(sha(Path(p))==h for p,h in c['files'].items())
science=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(Path(p))==h for p,h in science['files'].items())
pth=Path('/data/mlr_ifa/env/lib/python3.8/site-packages/zzz_mlr_lambda_live_collection_20261102.pth')
enabled=E/'HOOK_ENABLED.json'
assert str(pth)==c['global_startup_file'] and pth.parent==Path('/data/mlr_ifa/env/lib/python3.8/site-packages')
assert not pth.is_symlink() and not enabled.is_symlink()
assert sha(pth)==c['files'][str(pth)] and sha(enabled)=='44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a'
processes=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid() and (str(Q) in cmd or any(t in cmd for t in c['startup_targets'])):processes.append(dict(pid=int(p.name),cmd=cmd))
 except OSError:pass
assert not processes,processes
before={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in [enabled,pth]}
enabled.unlink();pth.unlink()
assert not enabled.exists() and not pth.exists()
assert all(sha(Path(p))==h for p,h in science['files'].items())
out=dict(status='PASS_OWNED_HOOK_CLEANUP_AFTER_ACCEPTANCE',observed=time.time(),binding=BINDING,removed=before,remaining_target_processes=[],science_contract_unchanged=True,optimizer_calls=0,environments=0,other_files_deleted=0)
(Q/'OWNED_HOOK_CLEANUP.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
