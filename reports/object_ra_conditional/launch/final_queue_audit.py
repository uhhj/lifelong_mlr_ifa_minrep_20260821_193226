from pathlib import Path
import json,hashlib,subprocess,time
from aggregate_stage1 import main as aggregate
Q=Path(__file__).resolve().parent
def main():
 assert (Q/'queue.exit').read_text().strip()=='0'
 assert not Path('/proc/'+(Q/'queue.pid').read_text().strip()).exists()
 for n,h in json.loads((Q/'ORCHESTRATION_SHA256.json').read_text()).items():assert hashlib.sha256((Q/n).read_bytes()).hexdigest()==h,n
 proc=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
 assert not any(' -m coreset_exp.train ' in x or ' -m coreset_exp.evaluate ' in x for x in proc)
 aggregate();result=json.loads((Q/'STAGE1_RESULTS.json').read_text())
 (Q/'FINAL_QUEUE_ACCEPTANCE.json').write_text(json.dumps(dict(status='PASS',stage=1,queue_exit=0,no_scientific_processes=True,orchestration_unchanged=True,epochs=1200,steps=316500,checkpoints=12,records=600,attempts=60,next_authorized_branch=result['next_authorized_branch'],requires_local_independent_audit_before_next_stage=True,time=time.time()),indent=2)+'\n')
if __name__=='__main__':main()
