import json,time,os
from pathlib import Path
from coreset_exp.common import O,S,SEED,sha,write,source_hashes
def main():
    assert not (O/'CONTRACT.json').exists()
    assert json.loads((O/'OFFLINE_ACCEPTANCE.json').read_text())['status']=='PASS'
    binding=json.loads((O/'PARENT_BINDING.json').read_text());P=Path(binding['parent']);pc=json.loads((P/'CONTRACT.json').read_text())
    files={str(p):sha(p) for p in [O/'OFFLINE_ACCEPTANCE.json',O/'PARENT_BINDING.json',O/'resolved_ra_coreset.json',O/'resolved_eval.json',O/'FEATURES.json',O/'MODAL_CONTRACT.json',*sorted((O/'bindings').glob('*.json')),*sorted((O/'offline_selection').glob('*.json')),*sorted(S.parent.glob('*.py')),*sorted(S.parent.glob('*.sh')),S.parent/'PROTOCOL.md',S.parent/'ACTUAL_SPEC.yaml']}
    files.update(pc['inputs']);files.update(binding['protected'])
    queue=S.parent.parent;orchestration=json.loads((queue/'ORCHESTRATION_SHA256.json').read_text())
    for name,h in orchestration.items():assert sha(queue/name)==h,name
    files.update({str(queue/name):h for name,h in orchestration.items()})
    files[str(queue/'ORCHESTRATION_SHA256.json')]=sha(queue/'ORCHESTRATION_SHA256.json')
    write(O/'CONTRACT.json',dict(version='RA_GOAL_extension_legacy_protocol_v1',seed=SEED,source_sha256=source_hashes(),files=files,modal_sha256=binding['modal_sha256'],parent_contract=binding['contract_sha256'],epochs=400,optimizer_steps=76100,final_checkpoints=4,formal_records=200,vector_attempts=20,progression=dict(AUC='strictly greater than same-seed Coreset',FWT='not less than same-seed Coreset',OldFinal='not less than same-seed Coreset'),created=time.time()))
    write(O/'RUN_STATUS.json',dict(status='FROZEN_AWAITING_PREFLIGHT',time=time.time()))
if __name__=='__main__':main()
