"""Read existing illustrative RGB only; never run a simulator or model."""
from pathlib import Path
import hashlib,json,struct,zlib,shutil
import h5py
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'assets_local';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def png(rgb,path):
    h,w,c=rgb.shape;assert c==3
    chunk=lambda typ,b:struct.pack('>I',len(b))+typ+b+struct.pack('>I',zlib.crc32(typ+b)&0xffffffff)
    raw=b''.join(b'\0'+rgb[y].tobytes() for y in range(h))
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))
file=ROOT/'local_assets/libero50_official/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5'
acq=json.loads(file.with_suffix('.acquisition.json').read_text(encoding='utf-8-sig'))
assert acq['status']=='VERIFIED' and file.stat().st_size==acq['bytes']
actual_sha=sha(file);assert actual_sha==acq['sha256']
manifest={'local_only':True,'new_rollouts':0,'model_calls':0,'frame_processing':'Lossless RGB export; no enhancement, crop or synthesized content.','expert':[],'success':[]}
with h5py.File(file,'r') as f:
    data=f['data/demo_0/obs/agentview_rgb']
    for i,step in enumerate([0,len(data)//2,len(data)-1]):
        rgb=data[step];p=OUT/f'expert_{i}.png';png(rgb,p)
        manifest['expert'].append(dict(file=p.name,source=str(file.relative_to(ROOT)),source_sha256=actual_sha,dataset='data/demo_0/obs/agentview_rgb',frame_index=step,png_sha256=sha(p),dimensions=list(rgb.shape),role='Official LIBERO_10 demonstration illustration, not current GOAL/OBJECT benchmark result'))
base=ROOT/'execution/20261026_thesis_preliminary'
prov=json.loads((base/'REAL_FRAME_PROVENANCE.json').read_text(encoding='utf-8-sig'))
records=[r for r in prov['images'] if r['camera']=='agentview_image']
for i,r in enumerate(records):
    src=base/'assets_private'/r['name'];assert sha(src)==r['png_sha256']
    p=OUT/f'success_{i}.png';shutil.copy2(src,p)
    manifest['success'].append(dict(file=p.name,source=str(src.relative_to(ROOT)),png_sha256=sha(p),step=r['step'],run=prov['run'],task=prov['task_language'],success=prov['success'],role='Archived ALIGN successful policy rollout illustration; not latest DEBR result'))
manifest['usage']='Illustrative sources for the two experience types. Different tasks, not a paired performance comparison. RGB assets and image-bearing figures remain local; publish only code and lightweight provenance.'
(HERE/'ASSET_PROVENANCE.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'verified':True,'expert_frames':3,'success_frames':3,'new_rollouts':0}))
