"""Read-only asset availability and immutable source binding. No model forward."""
import json,os,hashlib,subprocess,time,ast
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
out={}
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU has active work'
for name,c in CASES.items():
    q=Path(c['source_root']);root=Path(c['controller_root']);private=Path(c['private']);stage=c['stage']
    assert (root/'controller.exit').read_text().strip()=='0'
    contract=js(q/'CONTRACT.json');assert all(sha(p)==h for p,h in contract['files'].items())
    text={n:(q/n).read_text() for n in c['implementation']}
    for n,s in text.items():assert hashlib.sha256(s.encode()).hexdigest()==c['implementation'][n],(name,n)
    def literal(n,key):
        for node in ast.parse(text[n]).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in node.targets):
                return ast.literal_eval(node.value.args[0] if isinstance(node.value,ast.Call) else node.value)
        raise AssertionError(key)
    parent=Path(literal('runtime.py','O'));source=Path(literal('runtime.py','S'));pc=js(parent/'CONTRACT.json')
    assert all(sha(source/n)==h for n,h in pc['source_sha256'].items())
    cp=private/('stage%d.pth'%(stage-1));stage_meta=js(q/('STAGE_S_%d.json'%(stage-1)))
    assert sha(cp)==stage_meta['sha256']
    caches={};files={str(cp):sha(cp),str(q/'CONTRACT.json'):sha(q/'CONTRACT.json'),str(parent/'CONTRACT.json'):sha(parent/'CONTRACT.json')}
    for task in range(6,stage):
        mp=private/'memory'/('task%d.pth'%task);meta_path=q/('MEMORY_S_%d.json'%task);meta=js(meta_path)
        assert sha(mp)==meta['cache_sha256'];teacher=private/('stage%d.pth'%task);assert sha(teacher)==meta['source_sha256']
        caches[str(task)]=dict(path=str(mp),meta_path=str(meta_path),metadata=meta,teacher=str(teacher))
        for p in [mp,meta_path,teacher]:files[str(p)]=sha(p)
    manifest=parent/('ra_coreset/dataset_manifests/task%d.json'%stage);files[str(manifest)]=sha(manifest)
    out[name]=dict(**c,parent=str(parent),source=str(source),parent_source_sha256=pc['source_sha256'],checkpoint=str(cp),checkpoint_steps=stage_meta['steps'],files=files,caches=caches,pop=literal('layout.py','POP'),valid=literal('layout.py','VALID'),old=literal('layout.py','OLD'))
capacity={}
for mount,bound,reserve,root in [('/root',0,1024**3,False),('/data',8*1024**3,2*1024**3,True)]:
    st=os.statvfs(mount);free=(st.f_bfree if root else st.f_bavail)*st.f_frsize;assert free-bound>=reserve
    capacity[mount]=dict(free=free,bound=bound,reserve=reserve)
print(json.dumps(dict(status='PASS_READONLY_ASSETS',observed=time.time(),cases=out,capacity=capacity,hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True),new_updates=0,new_environments=0)))
