"""Independent reconstruction of the archived training input and the real training sampler."""
import hashlib,sys,json,ast
from pathlib import Path
from decimal import Decimal
import numpy as np,cv2,torch
work,old,prev,edge,frozen,last=map(Path,sys.argv[1:7])
outdir=Path(__file__).resolve().parent
sys.path.insert(0,str(last))
from common import subjects,read,rp,xyz,recorded,visual_step
from phases import classify
profile=read(frozen/'candidate.json');code=(work/profile['adapter']).parent;root=code.parent
cache=root/'ode_cache';live=work/'piper_test_console_pika_b1_full_v1'
sys.path[:0]=[str(code),str(root)]
from train_task import TrainingFrames
from ode_policy_adapter import build_policy
from select_b1 import files,vec,roles,B
from verify_task import read_arrays
from ode_wrist_alignment import GroundedODE
def write(name,obj):
    dst=outdir/name
    tmp=dst.with_suffix('.tmp');tmp.write_text(json.dumps(obj,separators=(',',':'),allow_nan=False));tmp.replace(dst)
def source_transform():
    # Use the training cache builder's actual CHW -> square/padded RGB -> model pixels statements.
    tree=ast.parse((root/'lerobot_io.py').read_text())
    loop=next(v for v in ast.walk(next(n for n in tree.body if getattr(n,'name','')=='adapt')) if isinstance(v,ast.For) and isinstance(v.target,ast.Name) and v.target.id=='role')
    nodes=[v for v in loop.body if not (isinstance(v,ast.If) and 'j in' in ast.unparse(v.test))]
    fn=ast.parse('def to_cache(val):\n    pass').body[0]
    fn.body=[n for n in nodes if not (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='val')]
    fn.body[-1]=ast.Return(fn.body[-1].value)
    scope={'cv2':cv2,'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module([fn],[])),'archived_lerobot_io.adapt','exec'),scope)
    return scope['to_cache']
to_cache=source_transform()
def train_rgb(path):
    bgr=cv2.imread(str(path));assert bgr.shape==(480,640,3)
    # Archived LeRobot dtype=image uses RGB. Verify the reconstruction against the persisted cache.
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return to_cache(rgb.transpose(2,0,1))
def episode_arrays(rec,upto):
    d=rec['folder'];rr=rec['rows'];raw=[json.loads(line) for line in (d/'alignment.jsonl').read_bytes().splitlines()]
    first=next(i for i,v in enumerate(raw) if v['sample_monotonic_ns']==rr[0]['sample_monotonic_ns'])
    assert rr[:upto]==raw[first:first+upto]
    sub=raw[first:first+upto]
    im={};paths={}
    for role,key in zip(roles,('head_camera','wrist_camera')):
        paths[role]=files(d,role)[first:first+upto]
        assert len(paths[role])==len(sub)
        pixel=np.stack([train_rgb(p) for p in paths[role]])
        im[key]=torch.from_numpy(pixel).cuda()
    # Missing rows precede or fall outside scored windows. They never form inputs to a scored batch.
    good=[v.get('target') is not None and v.get('feedback') is not None for v in sub]
    def field(k):
        out=np.zeros((len(sub),7),np.float32)
        usable=[i for i,v in enumerate(sub) if v.get(k) is not None]
        out[usable]=vec([sub[i] for i in usable],k)
        return out
    states=field('feedback');commands=field('target');anchors=np.vstack([states[:1],commands[:-1]])
    inst=TrainingFrames.__new__(TrainingFrames);inst.n=len(sub)
    inst.x=dict(**im,state=torch.from_numpy(states).cuda(),command=torch.from_numpy(commands).cuda(),action=torch.from_numpy(anchors).cuda())
    inst.start=torch.zeros(len(sub),device='cuda',dtype=torch.long);inst.end=inst.start+len(sub)-1
    inst.history=torch.arange(-7,1,device='cuda');inst.future=torch.arange(8,device='cuda')
    ts=np.array([r['sample_monotonic_ns'] for r in sub],np.int64)
    file_error=max(abs(int(Decimal(p.stem)*10**9)-int(t)) for paths_ in paths.values() for p,t in zip(paths_,ts))
    return inst,sub,good,file_error
def init():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    cp=work/profile['checkpoint'];ident=read(last/'identity_check.json')
    field=read(live/'profiles.json')['B1-ode50-wrist-20261127']
    assert field['sha256']==ident['field'] and rp.sha(cp)==ident['ode']
    station=GroundedODE(field)
    saved=torch.load(cp,map_location='cpu',weights_only=True)
    train=build_policy(saved['stats'],saved['config'])
    train.load_state_dict(saved['ema'],strict=True)
    train=train.cuda().eval()
    assert saved['manifest_sha256']==rp.sha(root/'selection.json')
    return train,station,saved['stats']
def layer_delta(a,b):return float((a-b).abs().max().item())
def output_xyz(station,plan):return xyz(station,plan.detach().float().cpu().numpy())
def maxes(d,**vals):
    for k,v in vals.items():d[k]=max(d.get(k,0),float(v))
