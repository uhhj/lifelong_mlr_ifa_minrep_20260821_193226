"""Portable read-only verification of copied evidence; not experimental acceptance."""
import json,hashlib,sys
from pathlib import Path
R=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(R/'attachment/zr_v2_seed43_plan'))
from compare_metrics import load_records,metrics,LOCKED
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    checked=0
    for evidence in sorted(R.glob('*evidence*')):
        if not evidence.is_dir() or not (evidence/'PACKAGE_MANIFEST.json').exists():continue
        m=json.loads((evidence/'PACKAGE_MANIFEST.json').read_text())
        for n,row in m['files'].items():
            p=evidence/n;assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],n;checked+=1
        for name in ['ADMISSION.json','INTEGRATION.json','COST_ISOLATION.json']:assert json.loads((evidence/name).read_text())['status']=='PASS'
        assert len((evidence/'integration_calls.jsonl').read_text().splitlines())==4
        assert (evidence/'software_tests.exit').read_text().strip()=='0'
        c=json.loads((evidence/'CONTRACT.json').read_text())
        for f in (R/'source').rglob('*.py'):
            remote='/root/mlr_ifa/experiment_workspaces/ZRv2_OBJECT_seed43_20261018/source/'+str(f.relative_to(R/'source')).replace('\\','/')
            assert sha(f)==c['files'][remote],str(f)
        rows=json.loads((evidence/'epochs.json').read_text()) if (evidence/'epochs.json').exists() else []
        assert len({(r['task'],r['epoch']) for r in rows})==len(rows)
        assert all(r['batches']==[157,227,309,362][r['task']-6] for r in rows)
        structures=[json.loads(s) for s in (evidence/'STRUCTURE.jsonl').read_text().splitlines()] if (evidence/'STRUCTURE.jsonl').exists() else []
        assert all(r['passed'] for r in structures)
    parent=R/'parent_evaluation';rates,records,_=load_records(parent,False);assert metrics(rates)==LOCKED
    package=R/'attachment/zr_v2_seed43_plan'
    for n,row in json.loads((package/'MANIFEST.json').read_text()).items():assert sha(package/n)==row['sha256']
    out=dict(status='PASS_LIGHT_EVIDENCE_ONLY',files_checked=checked,parent_records=len(records),parent_metrics={k:str(v) for k,v in metrics(rates).items()},formal_completion_not_claimed=True)
    (R/'LOCAL_LIGHT_VERIFICATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
