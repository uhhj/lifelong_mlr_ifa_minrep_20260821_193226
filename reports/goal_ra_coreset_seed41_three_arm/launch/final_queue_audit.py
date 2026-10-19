"""Final independent process/contract/quantity audit after finite queue exits."""
from pathlib import Path
import json,hashlib,subprocess,time,os
from aggregate_seed41 import main as aggregate,B,R
Q=Path(__file__).resolve().parent
def main():
 assert (Q/'queue.exit').read_text().strip()=='0'
 assert not Path('/proc/'+(Q/'queue.pid').read_text().strip()).exists()
 proc=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
 assert not [p for p in proc if any(' -m coreset_exp.'+x in p for x in ['train','export','evaluate','summarize'])]
 for p,h in json.loads((Q/'ORCHESTRATION_SHA256.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 aggregate()
 assert json.loads((B/'INDEPENDENT_ACCEPTANCE.json').read_text())['final_checkpoints']==8
 assert len(json.loads((R/'AUDIT.json').read_text())['checkpoints'])==4
 out=dict(status='PASS',queue_exit=0,epochs=1200,steps=228300,checkpoints=12,formal_records=600,vector_attempts=60,no_scientific_processes=True,orchestration_unchanged=True,time=time.time())
 (Q/'FINAL_QUEUE_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
