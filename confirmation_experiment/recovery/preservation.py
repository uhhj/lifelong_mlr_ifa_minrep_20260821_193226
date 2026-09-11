"""Check that completed data survive recovery; only declared tails may grow."""
import hashlib
from confirmation_experiment.common import ROOT,sha,read,write
from confirmation_experiment.recovery.common import AUDIT

def run():
    index=read(AUDIT/'PRESERVED.json')['files'];results=[]
    prefix='runs/fixed_kappa_confirmation_v2/formal/seed45/soft/'
    append={prefix+x for x in ('events.jsonl','run_001/paper_repro_train_curve.csv',
            'run_001/lr_protocol.csv','run_001/fixed_kappa_epochs.csv')}
    append.add('runs/fixed_kappa_confirmation_v2/formal/seed45/mlr/primary_ep99.log')
    metadata={prefix+x for x in ('TRAIN_PROGRESS.json','snapshot_index.json','instrumentation_timing.json')}
    for name,m in index.items():
        p=ROOT/name
        if name in metadata:
            saved=AUDIT/'original_evidence'/name
            assert sha(saved)==m['sha256'];kind='mutable runtime metadata: original copy verified'
        elif name in append:
            assert p.stat().st_size>=m['bytes']
            with p.open('rb') as f:actual=hashlib.sha256(f.read(m['bytes'])).hexdigest()
            assert actual==m['sha256'],('Old prefix changed',name)
            kind='original byte prefix preserved'
        else:
            assert sha(p)==m['sha256'] and p.stat().st_size==m['bytes'],('Completed evidence changed',name)
            kind='unchanged'
        results.append({'path':name,'result':kind})
    write(AUDIT/'RESULT_PRESERVATION.json',{'status':'PASS','files':len(results),'checks':results})
    return len(results)

if __name__=='__main__':run()
