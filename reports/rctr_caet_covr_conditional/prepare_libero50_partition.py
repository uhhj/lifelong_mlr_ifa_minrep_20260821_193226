"""Resolve the published LOTUS kitchen split against the pinned official dataset."""
import ast
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parent
source=R/"asset_sources/lotus_suite_task_map.py"
tree=ast.parse(source.read_text())
mapping=None
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="libero_task_map" for t in node.targets):
        mapping=ast.literal_eval(node.value)
assert mapping is not None
all_tasks=mapping["libero_90"]+mapping["libero_10"]
pre=[0,1,2,6,7,11,12,13,18,19,20,22,23,24,28,29,30,33,35,36,38,40,41,92,93]
stages=[[3,4,5,8,9],[10,14,15,16,17],[21,25,26,27,31],[32,34,37,39,42],[43,44,45,98,99]]
indices=pre+sum(stages,[])
assert len(indices)==len(set(indices))==50
manifest=json.loads((R/"LIBERO50_DOWNLOAD_MANIFEST.json").read_text())
files={Path(r["path"]).name.removesuffix("_demo.hdf5"):r for r in manifest["files"]}
assert {all_tasks[i] for i in indices}==set(files)
def task(index):
    name=all_tasks[index];f=files[name]
    assert name.startswith("KITCHEN_")
    suite="libero_90" if index<90 else "libero_10"
    assert f["path"]==suite+"/"+name+"_demo.hdf5"
    return dict(libero100_index=index,name=name,source_suite=suite,
                dataset_relative_path=f["path"],raw_sha256=f["lfs"]["oid"],
                bddl_relative_path=suite+"/"+name+".bddl",
                init_relative_path=suite+"/"+name+".pruned_init")
out=dict(status="PARTITION_METADATA_VERIFIED_NOT_TRAINED",
    source="https://github.com/UT-Austin-RPL/Lotus",
    source_commit="ce38726a6f2530b140eaba17b62ccdc849daafd8",
    task_map_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    partition_source_sha256=hashlib.sha256((R/"asset_sources/lotus_benchmark.py").read_bytes()).hexdigest(),
    upstream_revision=manifest["revision"],pretraining=[task(i) for i in pre],
    lifelong_stages=[[task(i) for i in stage] for stage in stages],
    pretraining_demonstrations_per_task=50,lifelong_demonstrations_per_task=10,
    stage_epochs=100,batch_size=10,gamma=.1,ifa_weight=.1,
    evaluation=dict(initial_states=list(range(20)),episodes_per_task_cell=20,
        evaluated_task_cells_per_seed=75,episodes_per_seed=1500,seeds=[41,42,43],
        total_formal_episodes=4500,base_tasks_excluded=True),
    weights_status="Official LIBERO50 pretrained checkpoint not located; no surrogate task weights allowed")
p=R/"libero50_preparation";p.mkdir(exist_ok=True)
(p/"PARTITION.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(dict(status=out["status"],pretraining_tasks=25,lifelong_tasks=25,matched_files=50)))
