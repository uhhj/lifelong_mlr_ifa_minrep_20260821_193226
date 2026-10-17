"""External audited recovery: normalize audit metadata keys only, not IFA tensors."""
import json,sys,time
from pathlib import Path
from zrg_exp.common import O,ROOT,S,sha,write,verify
import zrg_exp.train as train
E=Path(__file__).resolve().parent
verify()
contract=json.loads((E/'RECOVERY_CONTRACT.json').read_text())
assert contract['original_contract_sha256']==sha(O/'CONTRACT.json')
for name,value in contract['files'].items():assert sha(E/name)==value
original=train.refs
def normalized_refs(c,t):
    references,deviation=original(c,t)
    normalized={str(k):v for k,v in deviation.items()}
    assert len(normalized)==len(deviation) and normalized==json.loads(json.dumps(deviation))
    # The actual reference tensor dictionary is returned unchanged, with its original integer keys.
    assert all(type(k) is int for k in references)
    return references,normalized
train.refs=normalized_refs
if __name__=='__main__':train.main()
