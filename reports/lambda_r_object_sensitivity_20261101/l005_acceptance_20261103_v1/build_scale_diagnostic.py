"""Read-only CPU diagnostic for failed exact scale comparison; no policy forward."""
from pathlib import Path
import ast
A=Path(__file__).resolve().parent
t=ast.parse((A/'final_readonly.py').read_text(encoding='utf8'))
ss=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SCRIPTS')
tail='''
rows=[]
for stage in range(6,10):
 cache=torch.load(memory_path('S',stage),map_location='cpu');meta=js(Q/('MEMORY_S_%d.json'%stage))
 assert sha(memory_path('S',stage))==meta['cache_sha256']
 ids=[i for i,r in enumerate(manifests[stage]) if r['task']==stage and r['bc_valid']]
 y=torch.stack([torch.as_tensor(labels[stage][i]['next_action']) for i in ids]).float()
 now=torch.stack([y[:,lo:hi].square().mean().sqrt().clamp_min(.1) for lo,hi in [(0,3),(3,6),(6,7)]])
 stored=cache['scales'].cpu().numpy().astype(np.float32);actual=now.numpy()
 assert np.array_equal(stored,np.asarray(meta['scales'],np.float32))
 yy=y.numpy().astype(np.float64);ref=np.array([max(.1,np.sqrt(np.mean(yy[:,lo:hi]**2,dtype=np.float64))) for lo,hi in [(0,3),(3,6),(6,7)]])
 ulps=np.abs(stored.view(np.int32).astype(np.int64)-actual.view(np.int32).astype(np.int64))
 rows.append(dict(stage=stage,count=len(ids),stored=stored.tolist(),current=actual.tolist(),reference_float64=ref.tolist(),ulp_difference=ulps.tolist(),stored_reference_abs=np.abs(stored-ref).tolist(),current_reference_abs=np.abs(actual-ref).tolist(),cache_sha256=meta['cache_sha256'],cache_metadata_exact=True,source_float32_values_sha256=ah(y.numpy())))
print(json.dumps(dict(case=Q.name,scales=rows,optimizer_calls=0,environments=0,policy_forwards=0)))
'''
scripts={k:s[:s.index('collections=[]')]+tail for k,s in ss.items()}
wrapper='''
import subprocess,json
results={}
for case,s in SCRIPTS.items():
 p=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-'],input=s,text=True,capture_output=True)
 assert p.returncode==0,p.stderr
 results[case]=json.loads(p.stdout)
print(json.dumps(dict(status='READ_ONLY_SCALE_DIAGNOSTIC',cases=results,optimizer_calls=0,environments=0,policy_forwards=0)))
'''
p=A/'scale_diagnostic.py';p.write_text('SCRIPTS='+repr(scripts)+'\n'+wrapper,encoding='utf8');compile(p.read_text(encoding='utf8'),str(p),'exec')
