"""Check deployment and acquisition metadata; never equate it with experiment acceptance."""
import ast
import datetime
import hashlib
import json
import math
from pathlib import Path
R=Path(__file__).resolve().parent
P=R/"progress_20261021_server_queue"
def js(n):return json.loads((P/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    snapshot=js("SNAPSHOT_LATEST.json")
    native=js("LIBERO50_CPU_NATIVE.json")
    registry=js("LIBERO50_REGISTRY_ACCEPTANCE.json")
    handoff=js("CONTROLLER_V2_HANDOFF.json")
    revision=js("CPU_ENVIRONMENT_REVISION.json")
    assets=js("LIBERO50_SERVER_STATUS.json")
    partition=json.loads((R/"libero50_preparation/PARTITION.json").read_text())
    expected={r["dataset_relative_path"]:r for r in partition["pretraining"]+sum(partition["lifelong_stages"],[])}
    assert len(expected)==50
    assert assets["verified"]==len(assets["results"])<=50
    assert len({x["path"] for x in assets["results"]})==len(assets["results"])
    for x in assets["results"]:
        assert x["upstream_sha256"]==expected[x["path"]]["raw_sha256"]
        assert x["all_dataset_bytes_equal"] and x["all_attributes_equal"]
        assert x["status"]=="VERIFIED" and x["packed_bytes"]>0
    readiness=js("READINESS_20261021.json")
    full=js("LIBERO50_FULL_ASSET_AUDIT.json")
    effects=js("OBJECT_EFFECT_FEASIBILITY.json")
    assert assets["status"]=="DATA_VERIFIED" and assets["verified"]==50
    assert full["status"]=="DATA_AND_ENV_METADATA_READY" and full["verified"]==50
    assert (P/"LIBERO50_FULL_ASSET_AUDIT.exit").read_text().strip()=="0"
    assert len(full["rows"])==50 and all(x["dataset_verified"] for x in full["rows"])
    rows={x["name"]:x for x in full["rows"]}
    receipts={Path(x["path"]).stem.removesuffix("_demo"):x for x in assets["results"]}
    for name,row in rows.items():
        assert len(row["all_demo_lengths"])==len(row["demos"])==50
        assert row["full_windows"]==sum(row["all_demo_lengths"])
        assert row["lifelong_windows"]==sum(row["all_demo_lengths"][:10])
        assert row["packed_sha256"]==receipts[name]["packed_sha256"]
        assert row["initial_count"]>=20
    pretrain_steps=100*math.ceil(sum(rows[x["name"]]["full_windows"]
                                  for x in partition["pretraining"])/10)
    old=0;stage_counts=[]
    for stage,tasks in enumerate(partition["lifelong_stages"],1):
        current=sum(rows[t["name"]]["lifelong_windows"] for t in tasks)
        stage_counts.append(dict(stage=stage,current_windows=current,old_buffer_windows=old,
                                 batches_per_epoch=math.ceil((current+old)/10)))
        old+=sum(math.ceil(rows[t["name"]]["lifelong_windows"]/2) for t in tasks)
    main_steps=100*sum(x["batches_per_epoch"] for x in stage_counts)
    assert stage_counts==full["stage_counts"]==readiness["stage_counts"]
    assert pretrain_steps==full["shared_pretrain_steps"]==readiness["actual_shared_pretrain_steps"]
    assert main_steps==full["main_steps_per_seed"]==readiness["actual_main_steps_per_seed"]
    assert effects["status"]=="PASS" and set(effects["seeds"])=={"41","42","43"}
    assert all(v["pair_feasible"] and all(t["positive_pairs"]>0 for t in v["stages"].values())
               for v in effects["seeds"].values())
    assert native["status"]=="CPU_NATIVE_PASS_NOT_GPU_ADMITTED"
    assert native["environment_steps"]==native["optimizer_calls"]==native["cuda_allocated_bytes"]==0
    assert native["history_lengths"]==[1,4,8] and native["strict_brief_keys"]==100
    assert (P/"LIBERO50_FINAL_SOFTWARE.exit").read_text().strip()=="0"
    log=(P/"LIBERO50_FINAL_SOFTWARE.log").read_text()
    assert "Ran 14 tests" in log and "\nOK\n" in log
    assert "FROZEN_CONTRACT_INPUTS_PASS" in (P/"RCTR_HASH_CHECK.log").read_text()
    assert registry["status"]=="REGISTERED_GPU_GATED" and registry["total_entries"]==18
    assert registry["bundle_sha256"]==sha(R/"LIBERO50_SOURCE_BUNDLE.json")
    assert handoff["status"]=="PASS" and handoff["experiment_workers_preserved"]
    assert len(revision["changed"])==14 and "RCTR_OBJECT_42" in revision["preserved"]
    source=R/"libero50_additions"
    for p in source.rglob("*.py"):ast.parse(p.read_text(),filename=str(p))
    current=snapshot["runs"]["RCTR_OBJECT_42"]
    assert current["epochs"]==100 and current["optimizer_steps"]==15700
    assert current["formal_records"]==0
    stamp=datetime.datetime.fromtimestamp(snapshot["unix"],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    audit=dict(status="PARTIAL_METADATA_AUDIT_PASS_NOT_EXPERIMENT_COMPLETE",observed_at=stamp,
        source_bundle_sha256=registry["bundle_sha256"],registered_entries=18,
        software_tests=14,CPU_native_interface=True,LIBERO50_GPU_admitted=False,
        acquired_verified=assets["verified"],expected_files=50,
        all_data_environment_metadata_recomputed=True,shared_pretrain_steps=pretrain_steps,
        main_steps_per_seed=main_steps,total_libero50_optimizer_budget=pretrain_steps+3*main_steps,
        object_CAET_nonstatic_positive_pairs=True,
        rctr_frozen_input_check=True,active_run="RCTR_OBJECT_42",
        collection=current["collections"],formal_records=0,
        original_controller_preserved=True,new_controller_pid=handoff["new_pid"],
        metadata_files={p.name:sha(p) for p in P.glob("*") if p.suffix in (".json",".log",".exit")
                        and p.name not in ("AUDIT.json",)})
    (P/"AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
    status=dict(status="PARTIAL",observed_at=stamp,active="RCTR_OBJECT_42",phase="collect6",
        epochs=100,optimizer_steps=15700,formal_records=0,
        collection_branches=current["collections"][0]["branches"],
        object_entries_deployed=9,goal_entries_deployed=6,libero50_entries_deployed=3,
        libero50_verified_files=assets["verified"],libero50_total_files=50,
        libero50_training_implemented=True,libero50_gpu_admitted=False,
        libero50_shared_pretrain_steps=pretrain_steps,libero50_main_steps_per_seed=main_steps,
        libero50_pretraining_origin="new reproduction planned, not an official released checkpoint",
        caet_goal_admission="Stage7 has no moving cross-task positive pairs; not admitted",
        controller_version=2,controller_pid=handoff["new_pid"])
    (R/"RUN_STATUS.json").write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
