"""Run the common full-DEBR rollout on already-trained pure MLR weights."""
from pathlib import Path
import json,os,runpy,sys,time
def main():
 Q=Path(__file__).resolve().parent.parent;A=Q/'evaluation_unification_v1';R=Path(os.environ['REVERSE_CASE_ROOT'])
 sys.path.insert(0,str(A))
 import unified_mlr_runtime as rt
 rt.verify();inp=rt.INPUT
 assert (R/'train.exit').read_text().strip()=='0'
 native=Path(inp['output_root'])/'evaluation'
 assert not list(native.glob('trials*.jsonl')) and not list(native.glob('attempts*.jsonl')),'Old native evaluation must not already have run'
 assert not (R/'EVALUATION_STARTED.json').exists() and not (R/'NORMALIZED_OUTCOMES.json').exists()
 manifest=rt.js(Q/'EVALUATION_PROTOCOL_AMENDMENT.json')
 rt.write(R/'EVALUATION_PROTOCOL_DECLARATION.json',dict(status='DECLARED_BEFORE_ANY_FORMAL_EPISODE',time=time.time(),protocol=manifest['id'],method='MLR',training_contract_sha256=rt.sha(R/'CONTRACT.json'),manifest_sha256=rt.sha(Q/'EVALUATION_PROTOCOL_AMENDMENT.json'),source_sha256=manifest['evaluation_source_sha256'],extra_training_updates=0))
 runpy.run_path(str(A/'evaluate.py'),run_name='__main__')
 raw=rt.js(R/'OUTCOMES.json');assert len(raw)==200
 digest=rt.sha(R/'OUTCOMES.json')
 records=[dict(x,method='MLR',evaluation_protocol=manifest['id'],source_outcomes_sha256=digest) for x in raw]
 rt.write(R/'NORMALIZED_OUTCOMES.json',records)
 rt.write(R/'EVALUATION_PROTOCOL_COMPLETE.json',dict(status='PASS_COMMON_DEBR_EVALUATION_COMPLETE_PENDING_FINAL_AUDIT',protocol=manifest['id'],episodes=200,normalised_sha256=rt.sha(R/'NORMALIZED_OUTCOMES.json'),source_outcomes_sha256=digest,training_repeated=False,time=time.time()))
