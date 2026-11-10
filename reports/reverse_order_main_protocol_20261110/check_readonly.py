"""Read new-root lightweight progress; gate scalars are emitted as bounded shards."""
from pathlib import Path
import sys,json,time,subprocess
Q=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering04')
sys.path.insert(0,str(Q))
from reverse_common import read,sha,verify,capacity,processes
verify();light={};diagnostics={}
def add(path,name):
    raw=Path(path).read_bytes()
    if str(path).endswith('.jsonl'):raw=raw[:raw.rfind(b'\n')+1]
    import hashlib
    light[name]=dict(text=raw.decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
for name in ['STATUS.json','EVENTS.jsonl','FAILURE.json','controller.exit','LAUNCH.json','OBJECT_COMPLETE.json','GOAL_COMPLETE.json','FINAL_CPU_REQUEST.json']:
    if (Q/name).exists():add(Q/name,'coordinator/'+name)
for case,spec in read(Q/'SPEC.json')['cases'].items():
    r=Q/'cases'/case;inp=read(r/'INPUT.json')
    for p in r.iterdir():
        if not p.is_file():continue
        if p.name in ['INPUT.json','INPUT_COUNTS.json','CONTRACT.json','EXTERNAL_FILES.json','PREPARED_CASE.json','SELECTION_BINDING.json','TASK_MAPPING.json'] or p.name.startswith('EXPECTED_IFA_'):continue
        if p.name=='GATES.jsonl':
            rows=[json.loads(x) for x in p.read_text().splitlines() if x]
            for part,start in enumerate(range(0,len(rows),5000)):diagnostics[case+'_GATES_%03d'%part]=rows[start:start+5000]
        elif p.suffix in ['.json','.jsonl','.exit']:add(p,case+'/'+p.name)
    if spec['method']=='MLR':
        base=Path(inp['output_root'])
        for name in ['epochs.json','TRAINING_COMPLETE.json','INITIAL_STATE.json']:
            if (base/name).exists():add(base/name,case+'/native/'+name)
        for p in (base/'evaluation').glob('*.jsonl') if (base/'evaluation').exists() else []:add(p,case+'/native/evaluation/'+p.name)
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True)
print(json.dumps(dict(status='READONLY_REVERSE_PROGRESS_NOT_FINAL',observed=time.time(),contract_sha256=sha(Q/'CONTRACT.json'),processes=processes(),capacity=capacity(),gpu=gpu.stdout,light=light,diagnostics=diagnostics,new_optimizer=0,new_environments=0)))
