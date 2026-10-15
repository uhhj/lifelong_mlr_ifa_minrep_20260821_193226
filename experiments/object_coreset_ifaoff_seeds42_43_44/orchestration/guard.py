import json,hashlib,os
from pathlib import Path
from coreset_exp.common import O,S,verify
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
c=json.loads((O/'EVALUATION_CONTRACT.json').read_text());verify()
assert sha(O/'CONTRACT.json')==c['training_contract_sha256']
assert sha(O/'resolved_eval.json')==c['resolved_eval_sha256']
assert sha(O/'OLD_INPUTS_READONLY.json')==c['old_inputs_sha256']
for n,h in c['source_evaluation_sha256'].items():assert sha(S/'coreset_exp'/n)==h
print('TRAINING_AND_ORIGINAL_EVALUATION_CONTRACT_PASS',c['seed'])
