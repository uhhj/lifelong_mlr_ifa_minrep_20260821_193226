"""Run once after natural success; preserve failures instead of retrying science."""
from runtime import *
assert (Q/'controller.exit').read_text().strip()=='0' and not processes([Q,P])
verify()
with (Q/'FINAL_CPU_REQUEST.json').open('x') as f:json.dump(dict(pid=os.getpid(),observed=time.time(),status='RUNNING_NO_REPEAT'),f)
assert os.statvfs('/data').f_bavail*os.statvfs('/data').f_frsize>32*1024**2
stamp=time.strftime('%Y%m%d_%H%M%S');out=Q/('final_cpu_'+stamp);out.mkdir()
results={}
for case in CASES:
 e=env(case);e['CUDA_VISIBLE_DEVICES']='';e['OMP_NUM_THREADS']='1'
 p=subprocess.run([PY,'-B',str(Q/'final_case_cpu.py'),case],env=e,cwd=Q,capture_output=True,text=True)
 write(out/(case+'_ATTEMPT.json'),dict(exit=p.returncode,stderr=p.stderr,stdout=p.stdout))
 assert p.returncode==0,('CPU acceptance failed; preserve evidence',str(out),case,p.stderr[-2000:])
 result=json.loads(p.stdout.strip().splitlines()[-1]);assert result['status']=='PASS_CPU_CASE_RECONSTRUCTION';results[case]=result
verify();assert not processes([Q/'cases',P])
write(out/'CPU_RESULTS.json',dict(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',cases=results,optimizer_calls=0,environments=0,observed=time.time()))
print(json.dumps(dict(directory=str(out),results=results,status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT')))
