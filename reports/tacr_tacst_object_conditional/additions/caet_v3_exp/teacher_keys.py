import json,torch
from torch.nn import functional as F
from coreset_exp.common import O,sha,save,write,rng,restore_rng
from coreset_exp.model import cfg,strict_downstream
from libero.lifelong.models.brief import BriefPolicy
from .model import Student,hidden
def prepare_keys(a,t,ds):
 root=O/'teacher_keys';path=root/('stage%d.pth'%t);meta=path.with_suffix('.json')
 main=O/'ra_coreset/models'/('mlrifa_%d_model_ep99.pth'%(t-1));aux=O/'ra_coreset/aux_heads'/('task%d.pth'%(t-1))
 identity=dict(stage=t,teacher_stage=t-1,teacher_main_sha256=sha(main),teacher_aux_sha256=sha(aux),label_version='NEXT_ACTION_NEXT_EFFECT_GRIP_SIGN_PRE_PRUNE_V1',support_sha256=a.support_sha256)
 ids=[i for i,r in enumerate(ds.rows) if r['task']<t and r['bc_valid']]
 if path.exists():
  m=json.loads(meta.read_text());assert m['identity']==identity and m['sha256']==sha(path)
  keys=torch.load(path,map_location='cpu');assert list(keys)==ids
  return keys,dict(main=identity['teacher_main_sha256'],student=identity['teacher_aux_sha256'],keys=m['sha256'])
 saved=rng()
 try:
  teacher=BriefPolicy(cfg('ra_coreset'),None).cuda();assert strict_downstream(teacher,main)==100
  student=Student().cuda();student.load_state_dict(torch.load(aux,map_location='cpu')['model'],strict=True)
  teacher.eval();student.eval()
  for p in list(teacher.parameters())+list(student.parameters()):p.requires_grad_(False)
  keys={};batches=0
  with torch.no_grad():
   for start in range(0,len(ids),32):
    ix=ids[start:start+32];h=hidden(teacher,ds.features(ix));c=student.deploy.R(h)
    acts=torch.stack([ds.rows[i]['next_action'] for i in ix]).cuda();z=F.normalize(student.condition(c,acts),dim=-1)
    assert z.shape==(len(ix),128) and torch.isfinite(z).all() and torch.allclose(z.norm(dim=-1),torch.ones(len(ix),device=z.device),atol=1e-5)
    for i,k in zip(ix,z):keys[i]=k.detach().cpu().clone()
    batches+=1
  save(path,keys);write(meta,dict(identity=identity,sha256=sha(path),dimension=128,count=len(keys),teacher_forward_batches=batches,teacher_optimizer_calls=0,teacher_frozen=True,rows=[dict(index=i,task=ds.rows[i]['task'],sample=ds.rows[i]['sample'],demo=ds.rows[i]['demo'],u=ds.rows[i]['u'],sample_sha256=ds.rows[i]['sample_sha256']) for i in ids]))
  del teacher,student;torch.cuda.empty_cache()
 finally:restore_rng(saved)
 return keys,dict(main=identity['teacher_main_sha256'],student=identity['teacher_aux_sha256'],keys=sha(path))
