"""Bounded read-only SSH check; credentials exist only in getpass memory."""
import datetime
import getpass
import hashlib
import json
from pathlib import Path
import paramiko
import argparse
import shlex

R=Path(__file__).resolve().parent
Q="/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020"
COMMAND=f"/data/mlr_ifa/env/bin/python {Q}/monitor_snapshot_v2.py"
W="/data/mlr_ifa/experiment_workspaces/RCTR_OBJECT_seed42_full_20261021"
DETAIL_CODE=r'''
import collections,json,time,subprocess
from pathlib import Path
from coreset_exp.common import O,verify,sha
c=verify()
root=O/"rctr/task6"
rows=[json.loads(p.read_text()) for p in sorted((root/"committed").glob("*.json"))]
def read(p):
    return json.loads(p.read_text()) if p.exists() else None
collections_by_task={}
for task in (6,7,8):
    stage=O/"rctr"/("task%d"%task)
    if not stage.exists():continue
    committed=[read(p) for p in sorted((stage/"committed").glob("*.json"))]
    ledger=stage/"attempts.jsonl"
    raw=ledger.read_bytes() if ledger.exists() else b""
    lines=raw.splitlines(keepends=True)
    events=[json.loads(line) for line in lines if line.endswith(b"\n")]
    starts=[e["key"] for e in events if e["event"]=="branch_start"]
    ends=[e["key"] for e in events if e["event"]=="branch_complete"]
    collections_by_task[str(task)]=dict(committed=committed,
        acceptance=read(stage/"ACCEPTANCE.json"),status=read(stage/"RUN_STATUS.json"),
        live_step=read(stage/"LIVE_STEP.json"),
        events=dict(collections.Counter(e["event"] for e in events)),
        branch_starts=starts,branch_completes=ends,
        ledger_bytes=len(raw),ledger_sha256=sha(ledger) if ledger.exists() else None,
        incomplete_tail_lines=sum(not line.endswith(b"\n") for line in lines),
        sidecar_sha256=sha(stage/"sidecar.pth") if (stage/"sidecar.pth").exists() else None)
auxiliary={}
for task in (6,7,8,9):
    files=sorted((O/"ra_coreset/auxiliary").glob("task%d_epoch*.json"%task))
    if files:
        stats=[r for p in files for r in read(p)]
        auxiliary[str(task)]=dict(epoch_files=len(files),updates=len(stats),
            active_updates=sum(r["auxiliary"] is not None for r in stats),
            first_step=stats[0]["step"],last_step=stats[-1]["step"],
            all_steps=[r["step"] for r in stats],
            file_sha256={p.name:sha(p) for p in files})
evaluation=O/"ra_coreset/evaluation"
def jsonlines(p):
    return [json.loads(line) for line in p.read_bytes().splitlines(keepends=True) if line.endswith(b"\n")]
formal=dict(trials=[r for p in sorted(evaluation.glob("trials*.jsonl")) for r in jsonlines(p)],
    attempts=[r for p in sorted(evaluation.glob("attempts*.jsonl")) for r in jsonlines(p)],
    cells={p.name:read(p) for p in sorted(evaluation.glob("cell*.json"))},
    complete=read(evaluation/"EVALUATION_COMPLETE.json"))
workers=[]
for proc in Path('/proc').glob('[0-9]*'):
    try:
        env=(proc/'environ').read_bytes().split(b'\0')
        if ('CORESET_OUT='+str(O)).encode() in env:
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if 'python -c' not in cmd and 'python -u -c' not in cmd:
                workers.append(dict(pid=int(proc.name),command=cmd))
    except (OSError,PermissionError):pass
result=dict(unix=time.time(),status="FROZEN_CONTRACT_INPUTS_PASS",
    contract_sha256=sha(O/"CONTRACT.json"),source_files=len(c["source_sha256"]),
    committed=rows,collections_by_task=collections_by_task,auxiliary=auxiliary,
    epochs=read(O/"ra_coreset/epochs.json"),run_status=read(O/"RUN_STATUS.json"),
    training_complete=read(O/"ra_coreset/TRAINING_COMPLETE.json"),formal=formal,
    worker_processes=workers,
    exits={p.name:p.read_text().strip() for p in O.glob("*.exit")},
    collection_status=read(root/"RUN_STATUS.json"),
    live_step=read(root/"LIVE_STEP.json"),commands=read(O/"pipeline.pid.json"),
    logs={p.name:dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime,
         tail=p.read_text(errors="replace")[-2000:])
         for p in sorted(O.glob("*.log")) if p.name.startswith(("collect","train","evaluate","pipeline","supervisor"))},
    checkpoint_files=[dict(name=p.name,bytes=p.stat().st_size,mtime=p.stat().st_mtime)
                      for p in (O/"ra_coreset/models").glob("*.pth")])
queue=Path('/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020')
if (O/'PROCESS_EXIT_ACCEPTANCE.json').exists():
    parent=Path(read(O/'PARENT_BINDING.json')['parent'])
    names=('AUDIT.json','PROCESS_EXIT_ACCEPTANCE.json','TASK6_TENSOR_PARITY.json','GPU_PREFLIGHT.json')
    result['completion']={name:read(O/name) for name in names}
    result['completion']['audit_sha256']=sha(O/'AUDIT.json')
    result['completion']['receipt']=read(queue/'receipts/RCTR_OBJECT_42.json')
    result['completion']['checkpoints_sha256']={p.name:sha(p) for p in (O/'ra_coreset/models').glob('*ep99.pth')}
    result['completion']['baseline_trials']=[r for p in sorted((parent/'ra_coreset/evaluation').glob('trials*.jsonl')) for r in jsonlines(p)]
    result['completion']['baseline_audit']=read(parent/'AUDIT.json')
    result['completion']['initial_rng_equal']=read(O/'ra_coreset/INITIAL_STATE.json')==read(parent/'ra_coreset/INITIAL_STATE.json')
entry=read(queue/'REGISTRY.json').get('CAET_OBJECT_42')
if entry and Path(entry['output']).exists():
    output=Path(entry['output']);source=Path(entry['source']);contract=read(output/'CONTRACT.json')
    if contract:
        actual={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.sh') and '__pycache__' not in p.parts}
        assert actual==contract['source_sha256']
        assert all(sha(p)==h for p,h in contract['files'].items())
        feature_hashes=read(output/'FEATURES.json')['hashes']
        assert all(sha(output/'features'/name)==h for name,h in feature_hashes.items())
        aux={}
        for task in (6,7,8,9):
            files=sorted((output/'ra_coreset/auxiliary').glob('task%d_epoch*.json'%task))
            stats=[r for p in files for r in read(p)]
            aux[str(task)]=dict(epoch_files=len(files),updates=len(stats),active_updates=sum(r['effect_loss'] is not None for r in stats))
        result['caet']=dict(output=str(output),source=str(source),source_files=len(actual),source_input_feature_hashes_pass=True,
            contract_sha256=sha(output/'CONTRACT.json'),status=read(output/'RUN_STATUS.json'),
            epochs=read(output/'ra_coreset/epochs.json'),auxiliary=aux,
            launch=read(output/'LAUNCH_ACCEPTANCE.json'),preflight=read(output/'GPU_PREFLIGHT.json'),
            task6_parity=read(output/'TASK6_TENSOR_PARITY.json'),
            admission_exits={p.name:p.read_text().strip() for p in (source.parent/'admission').glob('*.exit')},
            phase_exits={p.name:p.read_text().strip() for p in output.glob('*.exit')},
            effects={str(p.relative_to(output)):read(p) for p in (output/'effects/models').glob('*/ACCEPTANCE.json')},
            effect_epochs={str(p.relative_to(output)):read(p) for p in (output/'effects/models').glob('*/epochs.json')},
            command_events=jsonlines(output/'commands.jsonl'),
            formal_records=sum(len(jsonlines(p)) for p in (output/'ra_coreset/evaluation').glob('trials*.jsonl')))
result['observed_end_unix']=time.time()
print(json.dumps(result,indent=2))
'''

CAET_CODE = r'''
import json,time,hashlib
from pathlib import Path
def read(p):
    return json.loads(p.read_text()) if p.exists() else None
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def jsonlines(p):
    return [json.loads(line) for line in p.read_bytes().splitlines(keepends=True) if line.endswith(b'\n')] if p.exists() else []
queue=Path('/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020')
result=dict(unix=time.time(),status='CAET_FROZEN_CONTRACT_INPUTS_PASS')
''' + DETAIL_CODE[DETAIL_CODE.index("entry=read(queue/'REGISTRY.json')"):DETAIL_CODE.index("result['observed_end_unix']")] + r'''
assert 'caet' in result
evaluation=output/'ra_coreset/evaluation'
result['caet']['formal']=dict(trials=[r for p in sorted(evaluation.glob('trials*.jsonl')) for r in jsonlines(p)],
    attempts=[r for p in sorted(evaluation.glob('attempts*.jsonl')) for r in jsonlines(p)],
    cells={p.name:read(p) for p in evaluation.glob('cell*.json')})
result['caet']['workers']=[]
for proc in Path('/proc').glob('[0-9]*'):
    try:
        env=(proc/'environ').read_bytes().split(b'\0')
        if ('CORESET_OUT='+str(output)).encode() in env:
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if 'python -c' not in cmd:result['caet']['workers'].append(dict(pid=int(proc.name),command=cmd))
    except (OSError,PermissionError):pass
result['caet']['checkpoints']=[dict(name=p.name,bytes=p.stat().st_size) for p in (output/'ra_coreset/models').glob('*.pth')]
parent=Path(read(output/'PARENT_BINDING.json')['parent'])
root=output/'ra_coreset'
result['caet']['initial_rng_equal']=read(root/'INITIAL_STATE.json')==read(parent/'ra_coreset/INITIAL_STATE.json')
result['caet']['storage_metadata']={}
for manifest in sorted((root/'storage').glob('task*.json')):
    task=int(manifest.stem[4:])
    if not (output/('retain%d.exit'%task)).exists():continue
    current=read(manifest);baseline=read(parent/'ra_coreset/storage'/manifest.name)
    assert current['K']==baseline['K']
    assert current['selection']['selected_ids']==baseline['selection']['selected_ids']
    def identities(m):return [(Path(x['source']).name,x['destination'],x['sha256']) for x in m['selected']]
    assert identities(current)==identities(baseline)
    assert all(sha(x['source'])==sha(root/'buffer'/x['destination'])==x['sha256'] for x in current['selected'])
    result['caet']['storage_metadata'][str(task)]=dict(count=current['K'],ordered_ids_equal=True,payloads_equal=True,
        manifest_sha256=sha(manifest),parent_manifest_sha256=sha(parent/'ra_coreset/storage'/manifest.name))
result['caet']['ifa_equal']={}
for f in root.glob('ifa_task*.json'):
    assert read(f)==read(parent/'ra_coreset'/f.name)
    result['caet']['ifa_equal'][f.name]=True
if (output/'PROCESS_EXIT_ACCEPTANCE.json').exists():
    result['caet']['completion']=dict(audit=read(output/'AUDIT.json'),
        process_exit=read(output/'PROCESS_EXIT_ACCEPTANCE.json'),audit_sha256=sha(output/'AUDIT.json'),
        receipt=read(queue/'receipts/CAET_OBJECT_42.json'),
        checkpoints_sha256={p.name:sha(p) for p in (root/'models').glob('*ep99.pth')},
        baseline_trials=[r for p in sorted((parent/'ra_coreset/evaluation').glob('trials*.jsonl')) for r in jsonlines(p)])
result['observed_end_unix']=time.time()
print(json.dumps(result,indent=2))
'''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--details",type=Path)
    ap.add_argument("--caet",type=Path,help="Only active CAET metadata; no repeated RCTR completion audit")
    ap.add_argument("--caet-seed",type=int,choices=(41,42,43),default=42)
    args=ap.parse_args()
    stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    assert not (args.details and args.caet)
    out=args.details or args.caet or R/("progress_"+stamp)
    if args.details or args.caet:
        assert out.resolve().parent==R and (out/"SNAPSHOT.json").is_file()
    client=paramiko.SSHClient()
    client.load_system_host_keys()
    # The destination key is already pinned by the user's previous native SSH connections.
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password=getpass.getpass("SSH password (memory only): ")
    try:
        client.connect("jq1.9gpu.com",port=11920,username="root",password=password,
                       allow_agent=False,look_for_keys=False,timeout=20)
        password=None
        if args.details or args.caet:
            caet_code=CAET_CODE.replace('CAET_OBJECT_42','CAET_OBJECT_%d'%args.caet_seed)
            command=(f"/data/mlr_ifa/env/bin/python -c {shlex.quote(caet_code)}" if args.caet else
                     f"source {W}/env.sh && /data/mlr_ifa/env/bin/python -c {shlex.quote(DETAIL_CODE)}")
            _,stdout,stderr=client.exec_command(command,timeout=180)
            raw=stdout.read();err=stderr.read();rc=stdout.channel.recv_exit_status()
            assert rc==0,(rc,err.decode(errors="replace"))
            details=json.loads(raw)
            assert details["status"]==("CAET_FROZEN_CONTRACT_INPUTS_PASS" if args.caet else "FROZEN_CONTRACT_INPUTS_PASS")
            prefix="CAET_DETAILS" if args.caet else "DETAILS"
            if args.caet and args.caet_seed!=42:prefix+='_seed%d'%args.caet_seed
            (out/(prefix+".json")).write_bytes(raw)
            (out/(prefix+"_RECEIPT.json")).write_text(json.dumps(dict(command=command,exit=rc,
                sha256=hashlib.sha256(raw).hexdigest(),server_writes=0),indent=2)+"\n")
            print(json.dumps(dict(status=details["status"],committed=len(details.get("committed",[])),
                                  directory=str(out))))
            return
        _,stdout,stderr=client.exec_command(COMMAND,timeout=60)
        raw=stdout.read();err=stderr.read();rc=stdout.channel.recv_exit_status()
        assert rc==0,(rc,err.decode(errors="replace"))
        snapshot=json.loads(raw)
        out.mkdir()
        (out/"SNAPSHOT.json").write_bytes(raw)
        with client.open_sftp() as sftp:
            for name in ("REGISTRY.json","controller_v2.pid.json","CONTROLLER_V2_HANDOFF.json",
                         "LIBERO50_REGISTRY_ACCEPTANCE.json"):
                with sftp.open(Q+"/"+name,"rb") as f:data=f.read()
                json.loads(data)
                (out/name).write_bytes(data)
        receipt=dict(status="READ_ONLY_SNAPSHOT",command=COMMAND,exit=rc,
                     snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                     unix=snapshot["unix"],writes_on_server=0,optimizer_calls=0,environment_steps=0)
        (out/"CHECK_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps(dict(directory=str(out),snapshot=snapshot),ensure_ascii=False,indent=2))
    finally:
        password=None
        client.close()

if __name__=="__main__":main()
