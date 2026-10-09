import json,hashlib,datetime,pathlib
L=pathlib.Path(__file__).resolve().parents[1];M=L/'monitor/20261008_2205';R=L/'recovery_import'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
t=load(M/'TRAINING_COMPLETE.json');v=load(M/'INPUT_VERIFICATION.json')
fail=(M/'failed_evaluate_primary.log').read_text(encoding='utf-8')
error_count=fail.count("ModuleNotFoundError: No module named 'libero.libero'")
assert error_count>0
addendum=dict(id='20261008_import_binding',created_local=datetime.datetime.now().isoformat(),
 original_contract_sha256='b82fc3d9a58d82302924c0b48eda56e95361227776d59f1f41505dda50b4ee31',
 original_contract_unchanged=True,original_frozen_source_unchanged=True,
 reason='Multiprocessing spawn cannot import sibling libero.libero before unpickling worker types. Failure before first env.seed completed; zero formal episodes and zero environment steps.',
 failed_attempt=dict(original_pipeline_exit=1,requested_spawn_workers=10,visible_import_error_traces=error_count,
  formal_records=0,environment_steps=0,constructed_environments='No successful construction recorded; zero inferred from identical spawn import failure, not from a per-worker construction ledger.'),
 fix='External sitecustomize applies exactly existing protective_e1.bindings.bind() to parent/spawned interpreters; original evaluator and scientific parameters unchanged.',
 validation='Import-only spawn child exit0, no environment or rollout. Full feature/checkpoint/pretrain/buffer hash audit PASS.',
 readonly_finalization_fix='External finalize_readonly.py permits only three regenerated .hydra output metadata hashes to differ, records actual hashes and verifies real code/config inputs. Original contract unchanged. Explicit finalizer.diff retained.',
 recovery_files={p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file()},
 resumed_process_pid=2536,
 command='nohup bash /data/mlr_ifa/experiments/E1_protective_object_seed42_20261008/engineering_recovery/20261008_import_binding/resume_eval.sh > /data/mlr_ifa/experiments/E1_protective_object_seed42_20261008/engineering_recovery/20261008_import_binding/resume.log 2>&1 < /dev/null &',
 successful_probes_exit=0,original_pipeline_exit=1,resumed_exit='pending')
(M/'RECOVERY_ADDENDUM.json').write_text(json.dumps(addendum,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
h=load(L/'MONITOR_HANDOFF.json');h['active_recovery']=addendum
h['recovery'].insert(0,'2026-10-08 22:08 recovery active: original pipeline.exit=1 is retained historical failure. Check engineering_recovery/20261008_import_binding/resume.pid, resume.exit, commands.log and live processes. Do NOT restart original pipeline. Training complete. Current script runs primary/secondary evaluation then external finalize_readonly only. Completion now requires resume.exit=0, no active recovery/evaluator process and full independent scientific acceptance; original pipeline.exit must stay1 as evidence.')
h['engineering_notes'].append('Known generated .hydra hash mismatch handled by separate finalize_readonly.py with explicit diff; no original contract/source rewrite. Read monitor/20261008_2205/RECOVERY_ADDENDUM.json. Original failure log remains O/evaluate_primary.log and R/failed_evaluate_primary.log; R/evaluate_primary.log is new recovery live output.')
h['latest_snapshot']=dict(status='TRAINED_EVALUATION_RECOVERED_RUNNING',epochs=400,optimizer_steps=105500,checkpoints=28,
 training_seconds=t['invocation_seconds'],original_pipeline_exit=1,recovery_exit=None,updated_local=datetime.datetime.now().isoformat())
(L/'MONITOR_HANDOFF.json').write_text(json.dumps(h,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(L/'RUN_STATUS.json').write_text(json.dumps(h['latest_snapshot'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(visible_import_error_traces=error_count,training_minutes=t['invocation_seconds']/60,recovery_files=len(addendum['recovery_files']))))
