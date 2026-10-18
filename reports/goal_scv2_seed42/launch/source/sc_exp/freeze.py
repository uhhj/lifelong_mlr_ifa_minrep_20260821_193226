import json,time,os
from pathlib import Path
from coreset_exp.common import O,S,sha,write,source_hashes
def main():
    assert not (O/'CONTRACT.json').exists()
    binding=json.loads((O/'PARENT_BINDING.json').read_text());P=Path(binding['parent']);pc=json.loads((P/'CONTRACT.json').read_text())
    files={str(p):sha(p) for p in [O/'PARENT_BINDING.json',O/'resolved_sc_coreset.json',O/'resolved_eval.json',O/'FEATURES.json',O/'MODAL_CONTRACT.json',*sorted((O/'bindings').glob('*.json')),*sorted((O/'offline_selection').glob('*.json')),*sorted(S.parent.glob('*.py')),*sorted(S.parent.glob('*.sh')),S.parent/'PROTOCOL.md']}
    files.update(pc['inputs']);files.update(binding['protected'])
    write(O/'CONTRACT.json',dict(version='SCv2_GOAL_seed42_v1',seed=42,source_sha256=source_hashes(),files=files,modal_sha256=binding['modal_sha256'],parent_contract=binding['contract_sha256'],epochs=400,optimizer_steps=76100,final_checkpoints=4,formal_records=200,vector_attempts=20,progression=dict(AUC='strictly greater than same-seed Coreset',OldFinal='not less than same-seed Coreset'),created=time.time()))
    write(O/'RUN_STATUS.json',dict(status='FROZEN_AWAITING_PREFLIGHT',time=time.time()))
if __name__=='__main__':main()
