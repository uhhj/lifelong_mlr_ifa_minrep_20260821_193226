"""Local prerequisite gate; never run before actual predecessor final acceptance."""
from pathlib import Path
import hashlib,json,subprocess,sys
R=Path(__file__).resolve().parent;old=R.parent/'20261110_s_stripped_mlr_ifa_align'
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
status=J(old/'RUN_STATUS.json');assert status['actual_final_cpu_acceptance'] is True
final=Path(status['evidence_directory']);a=J(final/'LOCAL_FINAL_AUDIT.json');e=J(final/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['formal']==2400 and a['checkpoints']==48
receipt=J(old/'PUBLICATION_RECEIPT.json');assert receipt['status']=='PUSH_VERIFIED'
G=R.parents[1]/'publish_mlr_branch';git='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
remote=subprocess.check_output([git,'-C',str(G),'ls-remote','origin','refs/heads/MLR'],text=True).split()[0]
# A later reviewed preparation publication may descend from the accepted report.
subprocess.run([git,'-C',str(G),'merge-base','--is-ancestor',receipt['commit'],remote],check=True)
manifest=J(old/'PUBLICATION_REVIEW.json')
name=str((final/'LOCAL_FINAL_AUDIT.json').relative_to(old)).replace('\\','/')
assert manifest['files_sha256'][name]['sha256']==sha(final/'LOCAL_FINAL_AUDIT.json')
assert J(R/'PREPARATION_LOCAL_AUDIT.json')['status']=='PASS_PREPARED_NOT_LAUNCHED'
cpu_text=json.dumps(dict(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',cases=e['cpu'],new_optimizer=0,new_environments=0),indent=2)+'\n'
auth=dict(predecessor_actual_final_status=a['status'],predecessor_final_audit_sha256=sha(final/'LOCAL_FINAL_AUDIT.json'),predecessor_publication_status=receipt['status'],predecessor_publication_commit=receipt['commit'],predecessor_remote_cpu_results=e['remote_cpu_directory']+'/CPU_RESULTS.json',predecessor_remote_cpu_results_sha256=hashlib.sha256(cpu_text.encode()).hexdigest(),prepared_contract=sha(R/'CONTRACT.json'))
assert not (R/'LAUNCH_AUTHORIZATION.json').exists(),'Do not blindly duplicate launch'
(R/'LAUNCH_AUTHORIZATION.json').write_text(json.dumps(auth,indent=2)+'\n')
(R/'launch_remote.py').write_text('AUTHORIZATION='+ascii(auth)+'\n'+(R/'launch_remote_template.py').read_text(),encoding='utf-8')
subprocess.run([sys.executable,'-X','utf8',str(R/'transport.py'),'launch_remote.py'],check=True)
