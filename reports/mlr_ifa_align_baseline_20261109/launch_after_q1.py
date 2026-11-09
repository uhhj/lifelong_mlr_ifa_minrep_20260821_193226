"""Bind real q1 final audits and published reports, then use one-shot deployment.
Usage: python launch_after_q1.py OBJECT_FINAL_DIR GOAL_FINAL_DIR
No effect without both real final acceptances and their verified MLR publication.
"""
from pathlib import Path
import json,hashlib,subprocess,sys
R=Path(__file__).resolve().parent;W=R.parents[1];OLD=R.parent/'20261109_support_calibration_rerun';G=W/'publish_mlr_branch'
GIT='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
def git(*args):return subprocess.check_output([GIT,'-C',str(G),*args])
def sha(raw):return hashlib.sha256(raw).hexdigest()
assert len(sys.argv)==3,'Supply actual OBJECT and GOAL final evidence directories'
assert git('branch','--show-current').decode().strip()=='MLR'
assert git('remote','get-url','origin').decode().strip()=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
head=git('rev-parse','HEAD').decode().strip();assert head==git('ls-remote','origin','refs/heads/MLR').decode().split()[0]
out=dict(status='PASS_BOTH_Q1_SUITES_ACCEPTED_AND_PUBLISHED',global_root='/data/mlr_ifa/experiment_workspaces/S_Q1_OBJECT_GOAL41_42_43_20261109_rerun_v1',global_contract='fbbce2917380d1cde1025f0006b6fdd6bc7959f92226bb5588cc431ee0c8d96c',suites={},remote_commit=head,published_files={})
def published(p):
 rel=p.resolve().relative_to(OLD.resolve()).as_posix();dest='reports/support_calibration_rerun_20261109/'+rel;raw=p.read_bytes()
 assert git('show',head+':'+dest)==raw,('Final evidence not published exactly',dest)
 out['published_files'][dest]=dict(sha256=sha(raw),bytes=len(raw));return dest
for suite,rawdir in zip(['OBJECT','GOAL'],sys.argv[1:]):
 d=Path(rawdir).resolve();d.relative_to((OLD/suite).resolve())
 p=d/'LOCAL_FINAL_AUDIT.json';raw=p.read_bytes();a=json.loads(raw)
 assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['epochs']==1200 and a['formal_episodes']==600
 c=json.loads((d/'PRIMARY_S_COMPARISON.json').read_text(encoding='utf8'));assert c['status'].startswith('PASS_') and c['source_Variant_final_sha256']==sha(raw)
 published(d/'PRIMARY_S_COMPARISON.json');published(d/'PRIMARY_S_COMPARISON.md')
 out['suites'][suite]=dict(audit=a,audit_text=raw.decode('utf8'),audit_sha256=sha(raw),publication_path=published(p))
published(OLD/'FINAL_REPORT.md')
bundle=json.loads((R/'DEPLOY_TEXT_BUNDLE.json').read_text(encoding='utf8'));assert 'POST_Q1_ACCEPTANCE.json' not in bundle
bundle['POST_Q1_ACCEPTANCE.json']=json.dumps(out,indent=2)+'\n'
# Keep the reviewed unarmed bundle intact; transport reads the armed file only in this invocation.
armed=R/'ARMED_DEPLOY_TEXT_BUNDLE.json';assert not armed.exists(),'Already armed; inspect original attempt before any retry'
armed.write_text(json.dumps(bundle),encoding='utf8')
sys.path.insert(0,str(R.parent/'20261027_support_gate_forks'))
import paramiko,remote
from credential_store import load
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());secret=load();assert secret
attempt=R/'DEPLOYMENT_ATTEMPT.json';assert not attempt.exists(),'Existing attempt; inspect instead of repeating'
attempt.write_text(json.dumps(dict(status='SUBMITTING_ONCE',root=out['global_root'],q1_publication=head)),encoding='utf8')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=secret,look_for_keys=False,allow_agent=False,timeout=20);secret=None
 script='BUNDLE='+ascii(bundle)+'\n'+bundle['deploy_remote.py']
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -B -',timeout=1800);i.write(script.encode());i.flush();i.channel.shutdown_write()
 stdout=o.read().decode();stderr=e.read().decode();code=o.channel.recv_exit_status()
 attempt.write_text(json.dumps(dict(exit=code,stdout=stdout,stderr=stderr,q1_publication=head),indent=2),encoding='utf8')
 assert code==0,stderr;print(stdout)
finally:secret=None;c.close()
