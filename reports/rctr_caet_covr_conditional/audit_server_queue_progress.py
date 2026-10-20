"""Check deployment and acquisition metadata; never equate it with experiment acceptance."""
import ast
import datetime
import hashlib
import json
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
        libero50_pretraining_origin="new reproduction planned, not an official released checkpoint",
        caet_goal_admission="Stage7 has no moving cross-task positive pairs; not admitted",
        controller_version=2,controller_pid=handoff["new_pid"])
    (R/"RUN_STATUS.json").write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
