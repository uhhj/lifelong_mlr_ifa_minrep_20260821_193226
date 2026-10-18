"""Run only after the finite pipeline has exited. No policy/environment execution."""
import os,sys,json,subprocess,time,hashlib
from pathlib import Path
O=Path('/root/mlr_ifa/experiments/ZRv2_OBJECT_seed43_20261018')
def main():
    assert (O/'pipeline.exit').read_text().strip()=='0'
    pid=int((O/'pipeline.pid').read_text());assert not (Path('/proc')/str(pid)).exists()
    matches=[]
    for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
        if 'python' in line and ('-m zr2_exp.calibrate' in line or '-m zr2_exp.evaluate' in line or '-m zr2_exp.audit' in line):matches.append(line)
    assert not matches,matches
    audit=json.loads((O/'AUDIT.json').read_text());assert audit['status']=='PASS'
    binding=json.loads((O/'PARENT_BINDING.json').read_text())
    for name,expected in binding['episode_hashes'].items():
        f=Path(binding['parent'])/'coreset/evaluation'/name
        assert hashlib.sha256(f.read_bytes()).hexdigest()==expected,('Parent episode file changed',name)
    for n in ['calibrate','evaluate','audit']:assert (O/(n+'.exit')).read_text().strip()=='0'
    out=dict(status='PASS',pipeline_exit=0,no_remaining_processes=True,time=time.time(),metric_gate=audit['metric_gate'])
    (O/'PROCESS_EXIT_ACCEPTANCE.json').write_text(json.dumps(out,indent=2))
    (O/'RUN_STATUS.json').write_text(json.dumps(dict(status='ELIGIBLE_FOR_REVIEW' if audit['metric_gate'] else 'AUDIT_PASS_METRIC_NOT_PASSED',engineering='COMPLETED',time=time.time()),indent=2))
    print(json.dumps(out))
if __name__=='__main__':main()
