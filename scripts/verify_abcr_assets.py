"""Read-only verification of explicitly listed experiment-A source assets.

No download, reconstructed data, model import or broad directory traversal.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    a=argparse.ArgumentParser();a.add_argument('--asset-root',type=Path,required=True)
    a.add_argument('--report-root',type=Path,required=True);args=a.parse_args()
    inputs=json.loads((args.report_root/'INPUT_MANIFEST.json').read_text())
    stages=json.loads((args.report_root/'STAGE_MANIFEST.json').read_text())['stages']
    source=Path(inputs['files'][0]['file']);oldroot=next(p for p in source.parents if (p/'features')==source.parents[2])
    errors=[];checked=0
    def check(path,expected,size=None):
        nonlocal checked
        q=args.asset_root/Path(path).relative_to(oldroot)
        if not q.is_file():errors.append(dict(file=str(q),error='MISSING'));return
        if (size is not None and q.stat().st_size!=size) or sha(q)!=expected:
            errors.append(dict(file=str(q),error='CONTENT_MISMATCH'))
        checked+=1
    for r in inputs['files']:check(r['file'],r['sha256'],r['size'])
    for c in inputs['configurations'].values():check(c['path'],c['sha256'])
    for s in stages:
        if s.get('seed')!=42 or s.get('status')!='READY':continue
        e=s['buffer_evidence'];check(e['plan'],e['plan_sha256'])
        check(str(Path(e['plan']).parent/'COMMITTED.json'),e['committed_sha256'])
        source_by_name={'t%d__%s'%(x['task'],Path(x['file']).name):x for x in inputs['files'] if x['seed']==42}
        for name in s['old_files']:
            x=source_by_name[name];check(str(Path(s['buffer_dir'])/name),x['sha256'],x['size'])
    print(json.dumps(dict(status='PASS' if not errors else 'FAIL',files_checked=checked,
        errors=errors,scope='Listed features, original configurations and surviving seed42 buffers only',
        public_archive=False),ensure_ascii=False,indent=2,allow_nan=False))
    raise SystemExit(bool(errors))


if __name__=='__main__':main()
