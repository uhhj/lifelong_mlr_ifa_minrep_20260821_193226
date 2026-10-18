"""Prepare reviewed lightweight files only; git commit/push are separate explicit commands."""
import pathlib,hashlib,json,shutil,re
R=pathlib.Path(__file__).resolve().parent
D=R.parents[1]/'publish_mlr_branch/reports/object_zrv2_seed43'
def main():
    assert not (D/'PUBLICATION_MANIFEST.json').exists(),'Preserve completed publication'
    chosen=[]
    for folder in ['source','attachment','parent_evaluation','launch_evidence','progress_evidence_154531']:
        chosen.extend(p for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in ['admission.py','pipeline.py','final_process_audit.py','monitor_snapshot.py','package_light.py','env.sh','verify_light.py','publish_prepare.py','EXECUTION_DECISIONS.md','MONITOR_HANDOFF.md','REPORT.md','source.patch','LOCAL_LIGHT_VERIFICATION.json']:chosen.append(R/name)
    allowed={'.py','.json','.jsonl','.md','.yaml','.txt','.sh','.patch','.log','.exit','.pid','.started'}
    manifest={}
    for p in chosen:
        assert p.suffix in allowed,p
        raw=p.read_bytes();assert len(raw)<15*1024*1024,p
        text=raw.decode('utf-8-sig')
        assert not re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{24,}',text),p
        assert not re.search(r'(?im)^\s*(?:password|passwd|api_key|access_token)\s*[:=]\s*[\x22\x27]?[^\s\x22\x27]{6,}',text),p
        rel=p.relative_to(R);dst=D/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert dst.read_bytes()==raw,('Preserve nonidentical existing file',dst)
        else:dst.write_bytes(raw)
        manifest[str(rel).replace('\\','/')]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    (D/'.gitattributes').write_text('* -text -whitespace\n')
    manifest['.gitattributes']=dict(bytes=(D/'.gitattributes').stat().st_size,sha256=hashlib.sha256((D/'.gitattributes').read_bytes()).hexdigest())
    (D/'PUBLICATION_MANIFEST.json').write_text(json.dumps(dict(status='PARTIAL',files=manifest),indent=2))
    print(json.dumps(dict(files=len(manifest),bytes=sum(x['bytes'] for x in manifest.values()),destination=str(D))))
if __name__=='__main__':main()
