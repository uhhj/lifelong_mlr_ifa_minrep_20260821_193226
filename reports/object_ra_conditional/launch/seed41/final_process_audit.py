from pathlib import Path
import os,json,time,subprocess
O=Path(os.environ['CORESET_OUT'])
def main():
    assert (O/'pipeline.exit').read_text().strip()=='0' and not (Path('/proc')/(O/'pipeline.pid').read_text().strip()).exists()
    lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
    bad=[x for x in lines if ' -m coreset_exp.' in x and any(v in x for v in ['train','evaluate'])];assert not bad,bad
    a=json.loads((O/'AUDIT.json').read_text());assert a['status']=='PASS'
    for n in ['train','evaluate','audit']:assert (O/(n+'.exit')).read_text().strip()=='0'
    out=dict(status='PASS',engineering='COMPLETED',metric_gate=a['metric_gate'],time=time.time(),pipeline_exit=0,no_remaining_processes=True)
    (O/'PROCESS_EXIT_ACCEPTANCE.json').write_text(json.dumps(out,indent=2))
    (O/'RUN_STATUS.json').write_text(json.dumps(dict(out,status='COMPLETED_PENDING_THREE_SEED_GATE'),indent=2));print(json.dumps(out))
if __name__=='__main__':main()
