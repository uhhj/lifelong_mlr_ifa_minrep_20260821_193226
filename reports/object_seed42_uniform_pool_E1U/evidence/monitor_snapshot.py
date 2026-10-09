"""Read-only progress/hash snapshot; no training/model/environment imports."""
from pathlib import Path
import json,csv,hashlib,time,subprocess,tarfile
from collections import Counter
O=Path('/data/mlr_ifa/experiments/E1U_uniform_pool_seed42_20261009')
S=Path('/data/mlr_ifa/experiment_workspaces/E1U_uniform_pool_20261009/source')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def lines(p):
    if not p.exists():return []
    text=p.read_text()
    return [json.loads(x) for x in text.splitlines() if x] if text.endswith('\n') else [json.loads(x) for x in text.splitlines()[:-1] if x]

def main():
    start=time.time();D=O/'monitor'/time.strftime('%Y%m%d_%H%M%S');D.mkdir(parents=True,exist_ok=False)
    contract=read(O/'CONTRACT.json')
    assert sha(O/'CONTRACT.json')=='59a4bdd042e50453d5c59195a54413a014725cfd2d91ddd6d5994999c0244143'
    code_bad=[k for k,v in contract['source_sha256'].items() if sha(S/k)!=v]
    old=read(O/'E1_READONLY_BEFORE.json');old_bad=[k for k,v in old.items() if sha(k)!=v]
    parity=read(O/'FEATURE_PARITY.json');feature_bad=[str(O/'features'/t/n) for t,fs in parity['features'].items() for n,h in fs.items() if sha(O/'features'/t/n)!=h]
    pretrain_hash=sha('/data/mlr_ifa/checkpoints/object_multitask_model_ep99.pth')
    assert not code_bad and not old_bad and not feature_bad
    assert pretrain_hash==contract['pretrain_sha256']
    training=read(O/'TRAINING_COMPLETE.json') if (O/'TRAINING_COMPLETE.json').exists() else None
    ck_bad=[k for k,v in training['checkpoints'].items() if sha(O/'run_001'/k)!=v] if training else []
    assert not ck_bad
    curve=list(csv.DictReader((O/'run_001/paper_repro_train_curve.csv').open()))
    epochs=Counter(int(r['stage']) for r in curve);steps=Counter()
    for r in curve:steps[int(r['stage'])]+=int(r['batches'])
    pairs=[read(p) for p in sorted((O/'pairing').glob('*.json'))]
    pair_bad=[p for p in pairs if not p['strict_pairing']]
    buffers={str(t):read(O/('buffer_pairing_task%d.json'%t)) for t in range(6,10) if (O/('buffer_pairing_task%d.json'%t)).exists()}
    identities=0;ref_counts=Counter();pools=0;numeric_fallback=[];prob_bad=[]
    for p in sorted((O/'selection_logs').glob('task*/epoch*.jsonl')):
        if '.interrupted_' in p.name:continue
        for row in lines(p):
            assert row['actual_samples'] is not None
            identities+=len(row['actual_samples'])
            r=row['refresh']
            if r:
                ref_counts[row['stage']]+=1
                if r['reason']!='OK' or any(r['invalid_candidates'].values()):numeric_fallback.append([row['stage'],r['step'],r['reason']])
                for t,probs in r['probabilities'].items():
                    pools+=1
                    if probs!=[1./len(probs)]*len(probs):prob_bad.append([row['stage'],r['step'],t])
    assert not prob_bad
    evaluations={}
    for view in ('primary','secondary'):
        rows=lines(O/('trials_'+view+'.jsonl'));attempts=lines(O/('attempts_'+view+'.jsonl'))
        events=Counter(r['event'] for r in attempts)
        keys=[(r['stage'],r['epoch'],r['task'],r['trial']) for r in rows]
        assert len(keys)==len(set(keys))
        evaluations[view]=dict(formal_records=len(rows),event_counts=dict(events),
            latest_attempt=attempts[-1] if attempts else None,
            complete_receipt=read(O/(view+'_COMPLETE.json')) if (O/(view+'_COMPLETE.json')).exists() else None)
    logmeta={n:dict(size=(O/n).stat().st_size,mtime=(O/n).stat().st_mtime) for n in ('train.log','evaluate_primary.log','evaluate_secondary.log') if (O/n).exists()}
    result=dict(captured_unix=time.time(),status=read(O/'RUN_STATUS.json'),epochs=dict(epochs),total_epochs=len(curve),steps=dict(steps),total_steps=sum(steps.values()),
        training=training,checkpoint_hashes_pass=not ck_bad,source_files_verified=len(contract['source_sha256']),original_files_verified=len(old),
        feature_files_verified=sum(len(fs) for fs in parity['features'].values()),pretrain_hash=pretrain_hash,
        pairing_epochs=len(pairs),pairing_mismatches=pair_bad,buffer_pairing=buffers,
        identity_rows=identities,refreshes=dict(ref_counts),task_pools=pools,uniform_probabilities_pass=not prob_bad,numeric_fallbacks=numeric_fallback,
        evaluations=evaluations,log_metadata=logmeta,
        recovery_metadata=dict(size=(O/'recovery/latest.pth').stat().st_size,mtime=(O/'recovery/latest.pth').stat().st_mtime),
        pipeline_exit=(O/'pipeline.exit').read_text().strip() if (O/'pipeline.exit').exists() else None,
        commands=(O/'commands.log').read_text(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        disk=subprocess.check_output(['df','-h','/data'],text=True),processes=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True),
        monitor_seconds=time.time()-start)
    (D/'SNAPSHOT.json').write_text(json.dumps(result,indent=2)+'\n')
    names=['RUN_STATUS.json','TRAINING_COMPLETE.json','commands.log','primary_COMPLETE.json','secondary_COMPLETE.json','trials_primary.jsonl','trials_secondary.jsonl','attempts_primary.jsonl','attempts_secondary.jsonl','eval_primary.log','eval_secondary.log','run_001/paper_repro_train_curve.csv','run_001/lr_protocol.csv']
    with tarfile.open(D/'light_evidence.tar.gz','w:gz') as tf:
        tf.add(D/'SNAPSHOT.json',arcname='SNAPSHOT.json')
        for name in names:
            if (O/name).exists():tf.add(O/name,arcname=name)
    print(json.dumps(dict(remote_snapshot=str(D),total_epochs=len(curve),total_steps=sum(steps.values()),checkpoints=len(training['checkpoints']) if training else 0,
        training_seconds=training['invocation_seconds'] if training else None,pairing_mismatches=len(pair_bad),refreshes=dict(ref_counts),identity_rows=identities,
        numeric_fallbacks=numeric_fallback,counts={k:v['formal_records'] for k,v in evaluations.items()},hashes='PASS',pipeline_exit=result['pipeline_exit'],monitor_seconds=result['monitor_seconds']),indent=2))
if __name__=='__main__':main()
