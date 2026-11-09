"""CPU tensor inspection only. No model forward, optimizer, environment or writes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,hashlib,datetime
from pathlib import Path
import torch
torch.set_num_threads(1)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def delta(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:return dict(comparable=False,shape_q1=list(a.shape),shape_s=list(b.shape))
    x=(a.double()-b.double()).abs()
    return dict(comparable=True,exact=torch.equal(a,b),elements=a.numel(),different=int((a!=b).sum()),max_abs=float(x.max()) if x.numel() else 0.,l2=float(x.square().sum().sqrt()))
before={p:sha(p) for p in EXPECTED}
assert before==EXPECTED,'Accepted file SHA changed'
out=dict(status='PASS_CPU_READONLY_PAIR_COMPARISON',observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),torch_version=torch.__version__,optimizer_calls=0,environment_calls=0,model_forward_calls=0,stages={},files=before)
for t in range(6,10):
    pair={tag:torch.load(FILES[tag][str(t)]['cp'],map_location='cpu')['state_dict'] for tag in ['Q1','S']}
    assert pair['Q1'].keys()==pair['S'].keys()
    cp={k:delta(pair['Q1'][k],pair['S'][k]) for k in pair['Q1']}
    del pair
    mm={tag:torch.load(FILES[tag][str(t)]['cache'],map_location='cpu') for tag in ['Q1','S']}
    maps={tag:{(x['trajectory'],x['step']):x for x in m['rows']} for tag,m in mm.items()}
    common=sorted(maps['Q1'].keys() & maps['S'].keys());rows=[];frames=[]
    for identity in common:
        a,b=maps['Q1'][identity],maps['S'][identity]
        row=dict(trajectory=identity[0],step=identity[1],initial_index=a['initial_index'],length_q1=a['length'],length_s=b['length'],mode_equal=a['mode']==b['mode'])
        for k in ['feature','mu','sd']:row[k]=delta(a[k],b[k])
        if t==6 and a['length']==b['length'] and a['feature'].shape==b['feature'].shape:
            for i in range(a['length']):frames.append(dict(trajectory=identity[0],absolute_step=identity[1]-a['length']+1+i,**delta(a['feature'][i],b['feature'][i])))
        rows.append(row)
    scale=delta(mm['Q1']['scales'],mm['S']['scales'])
    scale.update(q1=mm['Q1']['scales'].tolist(),s=mm['S']['scales'].tolist())
    out['stages'][str(t)]=dict(checkpoint_tensor_count=len(cp),checkpoint_equal=all(v['exact'] for v in cp.values()),checkpoint_different_count=sum(not v['exact'] for v in cp.values()),checkpoint_tensors=cp,cache_rows={k:len(m['rows']) for k,m in mm.items()},common_rows=len(common),q1_only=sorted(maps['Q1'].keys()-maps['S'].keys()),s_only=sorted(maps['S'].keys()-maps['Q1'].keys()),scales=scale,rows=rows,task6_saved_frames=frames)
out['files_unchanged_after']=all(sha(p)==h for p,h in before.items())
assert out['files_unchanged_after']
print(json.dumps(out))
