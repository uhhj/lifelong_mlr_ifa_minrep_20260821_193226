"""Replay a saved boundary to an existing anchor without creating scientific labels."""
import contextlib
import io
import time
import traceback
from gradient_relation_experiment.kernel import *
from gradient_relation_experiment.run import verify_code

OUT=ROOT/'reports/gradient_relation_4090_v2'


def run():
    verify_code();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'TRANSITION_PROBE.json').exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    source=ROOT/'runs/gradient_relation_v1/seed45'
    files={str(p.relative_to(ROOT)):sha(p) for p in source.iterdir() if p.is_file()}
    write(OUT/'PRESERVATION.json',{'time':time.time(),'source_files':files,'completed_points':24,
        'old_queue_process_absent':True,'preservation':'original directories are read-only; no records overwritten'})
    saved=torch.load(source/'RECOVERY.pt',map_location='cpu')
    checks={r['point']:r for r in (json.loads(x) for x in (source/'checks.jsonl').read_text().splitlines())}
    plan=read(ROOT/saved['plan_path']);assert sha(ROOT/saved['plan_path'])==saved['plan_sha256']
    target='s45_t8_e43_b247';expected=checks[target];target_step=43*plan['batches_per_epoch']+247
    cfg=train_config(45,OUT/'probe');ds=CachedWindows(plan['immutable_feature_origins'],plan['record_sha256'])
    sampler=torch.utils.data.DistributedSampler(ds,**{k:plan['sampler'][k] for k in ('seed','num_replicas','rank','shuffle','drop_last')})
    dl=torch.utils.data.DataLoader(ds,batch_size=10,sampler=sampler,num_workers=0)
    (OUT/'probe').mkdir(exist_ok=True);init_ddp(OUT/'probe/ddp_store')
    bootstrap_train(cfg);algo=build_train(cfg)
    with contextlib.redirect_stdout(io.StringIO()):algo.start_task(-1,dl)
    restore(algo,saved)
    start=capture(algo);restored=compare_states(saved,start);assert all(restored.values()),restored
    begin=time.monotonic();steps=0;actual=None
    for epoch in range(saved['epoch']+1,44):
        sampler.set_epoch(epoch);algo.policy.train()
        for i,b in enumerate(dl):
            stage_step=epoch*len(dl)+i
            if stage_step==target_step:
                actual={'start_model_sha256':digest(algo.policy.state_dict()),'start_optimizer_sha256':digest(algo.optimizer.state_dict())};break
            native(algo,b,8,epoch);steps+=1
        if actual is not None:break
        if epoch%5==0 and epoch>69:algo.policy.eval()
        algo.epoch_schedule_step(epoch)
    assert actual is not None
    match={k:expected[k]==v for k,v in actual.items()}
    write(OUT/'TRANSITION_PROBE.json',{'status':'PASS' if all(match.values()) else 'CROSS_HARDWARE_CONTINUATION_NOT_EQUIVALENT',
        'time':time.time(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
        'snapshot_restore_bitwise_equal':restored,'saved_boundary':{'task':8,'epoch':39,'step':12360},
        'target':target,'steps_replayed':steps,'matched':match,'expected':{k:expected[k] for k in actual},'actual':actual,
        'elapsed_seconds':time.monotonic()-begin,'scientific_points_added':0})
    for rel,h in files.items():assert sha(ROOT/rel)==h,('Original asset changed',rel)
    torch.distributed.destroy_process_group()


if __name__=='__main__':
    try:run()
    except Exception:
        write(OUT/'PROBE_FAILED.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
