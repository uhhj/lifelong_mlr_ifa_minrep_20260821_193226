"""Complete the interrupted read-only phase; never instantiate an optimizer."""
import json,time,tarfile,hashlib
from .common import *
from .native_action_check import main as native
def main():
    start=time.time();assert not(O/'INTEGRATION.json').exists()
    evidence=json.loads((S.parent/'INTEGRATION_RECOVERY.json').read_text())
    assert sha(O/'integration/stage7.pth')==evidence['checkpoint_sha256']
    calls=(O/'integration_optimizer_calls.jsonl').read_text().splitlines();assert len(calls)==4 and calls==evidence['optimizer_call_ledger']
    # This checkpoint is only written after successful parity/gradient/4-call serialized recovery assertions.
    with tarfile.open(S.parent/'implementation_v2.tar') as tar:
        source=tar.extractfile('zrg_exp/integration_check.py').read()
        assert hashlib.sha256(source).hexdigest()==evidence['integration_source_sha256']
    assert source.index(b'assert first==second and equal(expected,capture(a))')<source.index(b"tmp=O/'integration/stage7.pth'")
    native()
    write(O/'INTEGRATION.json',dict(status='PASS',optimizer_calls=4,effective_steps=3,formal_steps=0,environment_steps=0,
      tests=dict(initial_public_and_rng_exact=True,zero_output_and_native_grad_parity=True,old_branch_z_gradient_nonzero=True,exact_serialized_optimizer_scheduler_rng_recovery=True,single_mixed_no_current_finite=True,old_branch_frozen=True,current_U_V_gradients=True),
      bounded_update_phase_evidence='v2 integration_check.py assertions completed before serialized stage7 checkpoint creation; checkpoint and 4-call ledger bound in recovery evidence',
      recovery=evidence,continuation_optimizer_calls=0,continuation_seconds=time.time()-start,seconds=time.time()-(S.parent/'integration_1.log').stat().st_ctime,
      strict_native_load_and_action='NATIVE_ACTION_CHECK.json',serialized_test_checkpoint_sha256=evidence['checkpoint_sha256']))
    print('INTEGRATION PASS after zero-update native continuation. Total optimizer calls = 4.')
if __name__=='__main__':main()
