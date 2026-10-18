import json,shutil,subprocess,time
from .common import *
from .metrics import read,calculate,serial,EXPECTED
def main():
    assert not O.exists(),'Use a new run; never re-prepare existing contract'
    O.mkdir();ROOT.mkdir();(ROOT/'buffer').mkdir()
    parent=json.loads((P/'CONTRACT.json').read_text());passed=json.loads((PROTO/'PROTO_FINAL.json').read_text());assert passed['status']=='PASS_PROTOTYPE'
    for n,h in parent['source_sha256'].items():assert sha(S/n)==h,('parent source',n)
    inputs=json.loads((PROTO/'INPUT_SHA256.json').read_text())
    inputs.update(parent['inputs']);inputs.update(parent['configs'])
    fm=json.loads((P/'FEATURES.json').read_text())
    assert fm['contract_sha256']==sha(P/'CONTRACT.json') and fm['modal_sha256']==sha(P/'MODAL_CONTRACT.json')
    inputs.update({str(P/'features'/n):h for n,h in fm['hashes'].items()})
    for f in (P/'coreset').glob('buffer_after_task*.json'):inputs[str(f)]=sha(f)
    inputs[str(P/'RUN_STATUS.json')]=sha(P/'RUN_STATUS.json')
    for f in list(P.glob('resolved_*.json'))+[P/'CONTRACT.json',P/'MODAL_CONTRACT.json',P/'FEATURES.json']+list((P/'coreset/storage').glob('*.json'))+list((P/'coreset/evaluation').glob('*.jsonl'))+list((P/'coreset').glob('ifa_*.json'))+[P/'coreset/INITIAL_STATE.json']:
        inputs[str(f)]=sha(f)
    for f in (PROTO/'PROTO_FINAL.json',PROTO/'FINAL_ACCEPTANCE.json',PROTO/'residual.py',PROTO/'gate_core.py',PROTO/'geometry/gates.npz',PROTO/'instruction_bank.npz',PROTO/'SAMPLE_IDS.json'):
        inputs[str(f)]=sha(f)
    for p,h in inputs.items():assert sha(p)==h,p
    write(O/'INPUT_SHA256.json',inputs)
    for name in ('MODAL_CONTRACT.json','FEATURES.json'):shutil.copyfile(P/name,O/name)
    shutil.copyfile(S.parent/'spec.yaml',O/'spec.yaml');shutil.copyfile(S.parent/'PROTOCOL.md',O/'PROTOCOL.md')
    train=json.loads((P/'resolved_coreset.json').read_text());evaluation=json.loads((P/'resolved_eval.json').read_text())
    original=dict(train)
    train.update(run_root=str(ROOT),experiment_dir=str(ROOT/'models'),buffer_dir=str(ROOT/'buffer'),feature_dir=str(P/'features'))
    assert train['seed']==evaluation['seed']==SEED
    assert train['policy']['use_state'] is True and train['train']['batch_size']==10 and train['gamma']==.3
    write(O/'resolved_train.json',train);write(O/'resolved_eval.json',evaluation)
    (O/'features').symlink_to(P/'features',target_is_directory=True)
    counts,_=read(P/'coreset/evaluation')
    admission=json.loads((S.parent/'NUMERIC_ADMISSION.json').read_text());assert admission['status']=='PASS'
    write(O/'PARENT_BINDING.json',dict(upstream_commit='1ed6e0dd25cb5748ee5e887444619483bd9bdc31',actual_source_identity='parent CONTRACT source_sha256',path=str(P),contract_sha256=sha(P/'CONTRACT.json'),original_source_verified=True,baseline=serial(calculate(counts)),seed=SEED,baseline_counts={f"{s},{t}":v for (s,t),v in counts.items()}))
    write(O/'PROTOTYPE_BINDING.json',dict(path=str(PROTO),status=passed['status'],trained_residuals_imported=False,gate_sha256=sha(PROTO/'geometry/gates.npz'),source_sha256={n:sha(PROTO/n) for n in ('residual.py','gate_core.py')},acceptance_sha256=sha(PROTO/'PROTO_FINAL.json')))
    write(O/'GATES_AND_PROTECTION.json',dict(gates_sha256=sha(PROTO/'geometry/gates.npz'),protected={str(k):list(range(k)) for k in range(6,10)},future_instruction_gate_construction=False))
    model=dict(rank=8,U_shape=[75,8],V_shape=[8,512],raw_layout=['means35','scales35','logits5'],gate_dtype='float64',residual_dtype='float32',init='local CPU generator 42+task; U zero',trainable='shared temporal/public head/current UV',old_branch_jacobian_to_z=True,parent_modal_sha256=sha(O/'MODAL_CONTRACT.json'))
    write(O/'MODEL_CONTRACT.json',model)
    # Capacity includes the largest recovery atomic replacement, all finals and copied buffer.
    required=sum(p.stat().st_size for p in (P/'coreset/models').glob('*.pth'))+2*(P/'coreset/recovery/latest.pth').stat().st_size+sum(p.stat().st_size for p in (P/'coreset/buffer').glob('*.pt'))+160*1024**2
    free=shutil.disk_usage(O).free
    write(O/'REUSE_AUDIT.json',dict(status='PASS' if free>required else 'BLOCKED_SPACE',admission=admission,hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader']).decode(),historical_gpu=json.loads((P/'GPU_PREFLIGHT.json').read_text())['gpu'],disk_free=free,estimated_atomic_peak_bytes=required,parent_numeric_source='control_seed: cudnn deterministic true; fresh torch1.13 defaults, parent evaluate benchmark true',original_train_config=original,change_scope='Same seed42 residual/loss/model protocol; corresponding seed baseline/features/order; new output on user-authorized system disk',disclosed_gaps=['pretrain use_state unknown','single exploratory seed','global action RNG not per-trajectory isolated','historical driver not independently archived']))
    assert free>required,'Data disk cannot accommodate atomic peak'
    write(O/'RUN_STATUS.json',dict(status='PREPARED',formal_steps=0,formal_episodes=0))
    print('PREPARED',free,required)
if __name__=='__main__':main()
