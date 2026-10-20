"""Read-only provenance for a design memo; no model or experiment execution."""
from pathlib import Path
import hashlib,json,datetime
R=Path(__file__).resolve().parent;root=R.parents[1]
rels=[
 "execution/20261017_event_deep_audit/REPORT.md",
 "execution/20261017_tcc_zero_training/REPORT.md",
 "execution/20261017_update_coordination_diagnostic/REPORT.md",
 "execution/20261017_zr_coreset_seeds40_41/FAILURE_ANALYSIS.md",
 "execution/20261019_ra_object_conditional/analysis_20261020_zrv2/REPORT.md",
 "execution/20261020_zrv3_object_seed41/task6_readonly/REPORT.md",
 "execution/20261020_sazr_object_seed41/readonly_analysis/REPORT.md",
 "execution/20261020_gripper_repair_stageA/REPORT.md",
 "execution/20261020_czr_stageA/readonly_generalization/REPORT.md",
 "execution/20261013_coreset_object/source/libero/lifelong/models/policy_head.py",
 "execution/20261013_coreset_object/source/coreset_exp/export.py",
 "execution/20261020_all_methods_summary/ALL_RESULTS.json"]
manifest=[]
for rel in rels:
 p=root/rel;b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 manifest.append(dict(path=rel,bytes=len(b),sha256=h))
status=dict(status="DESIGN_ONLY",time=datetime.datetime.now().astimezone().isoformat(),
 source_files=manifest,training_started=False,model_loads=0,optimizer_calls=0,environment_calls=0,
 no_source_changes=True,novelty_status="Candidate design; not a comprehensive novelty proof",
 literature=[
  {"name":"DER","url":"https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html"},
  {"name":"CLEAR","url":"https://arxiv.org/abs/1811.11682"},
  {"name":"SPREAD","url":"https://arxiv.org/html/2603.08763v1#S3.SS3"},
  {"name":"R-Drop","url":"https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html"}])
(R/"SOURCE_AUDIT.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
(R/"RUN_STATUS.json").write_text(json.dumps({k:v for k,v in status.items() if k not in ("source_files","literature")},ensure_ascii=False,indent=2)+"\n",encoding="utf8")
print(json.dumps(dict(status="DESIGN_ONLY",source_hashes=len(manifest),training=0,rollouts=0)))
