"""Assemble the offline candidate exclusively in the diagnostic directory."""
from survey import *
import copy,hashlib,shutil
sys.path.insert(0,str(old))
from temporal import signature

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(root/n).write_text(json.dumps(x,sort_keys=True,allow_nan=False))
def build():
    cv2.setNumThreads(1)
    first_version='22473da5b736819b906b23561a960514916d95783c193c51cc22308347192030'
    if (root/'identity.json').is_file() and read(root/'identity.json')['profile']==first_version:
        for n in ('regression','archive','audit','identity','endpoints','misses'):
            if (root/(n+'.json')).exists():shutil.copyfile(root/(n+'.json'),root/('first_'+n+'.json'))
    profile=read(prev/'candidate.json')
    assert profile['sha256']=='6e92f46eba8f0eacf5707ba382c1e74dcf8af63e4c5c6af3571868cd360ba30a'
    alignment=profile['alignment']
    assert sha(prev/'ode_relative.py')==alignment['backend_sha256']
    for name,h in alignment['station_files'].items():
        assert sha(prev/name)==h
        shutil.copyfile(prev/name,root/name)
    sys.path.insert(0,str(work/'offline_diagnostics/b1_vision_contrasts_20261124'))
    from assess import reviewed
    ref=reviewed(work,work/'offline_diagnostics/b1_vision_contrasts_20261124')
    samples=[[],[]]
    for e,r in rows('train'):
        for i in (0,max(0,r['close']-18)):samples[0].append(signature(rp.imaging(r,i,'overhead')))
    for r in ref:
        if not r['train']:continue
        rec=rp.load_record(r['folder'],r['first'])
        ticks=[0] if r['close'] is None else [0,max(0,r['close']-18)]
        for i in ticks:samples[1].append(signature(rp.imaging(rec,i,'overhead')))
    prototypes=np.array([np.median(s,axis=0) for s in samples])
    limits=[]
    for k,s in enumerate(samples):
        ds=np.sqrt(np.mean((np.array(s)[:,None]-prototypes[None])**2,axis=2))
        limits.append(float(np.quantile(ds[:,k],.99)+3))
        print('signature training',k,len(ds),'correct',sum(np.argmin(ds,axis=1)==k),'P99+3',round(limits[-1],2),flush=True)
    model=rp.models()[1]
    from b1_alignment_gate import wrist_block,wrist_centered
    def envelope(records):
        heights=[];widths=[]
        for rec in records:
            c=rec['close']
            if c is None or not rp.valid(rec['rows'][c]):continue
            q=rp.at(rec['rows'][c],'feedback')
            tf=model.aligner.flange(np.rad2deg(q[:6]))
            tip=tf[:3,3]+tf[:3,:3]@model.aligner.par['tip']
            heights.append(float(tip[2]))
            b=wrist_block(rp.imaging(rec,c,'wrist'))
            if wrist_centered(b,q[6]):widths.append(b[2])
        assert len(heights)>=30 and len(widths)>=8
        return dict(max_height=float(np.percentile(heights,95)),min_face_width=float(np.percentile(widths,10)),
                    examples=len(heights),visible_examples=len(widths))
    envelopes=[envelope([r for _,r in rows('train')]),envelope([dict(**rp.load_record(r['folder'],r['first']),close=r['close']) for r in ref if r['train']])]
    print('expert close envelopes',envelopes,flush=True)
    write('view_geometry.json',dict(signatures=prototypes.tolist(),distance_limits=limits,close_envelopes=envelopes,
          archive_matrix=np.load(root/'private_scene_map.npz')['raw_matrix'].tolist()))
    s=(prev/'ode_relative.py').read_text()
    def edit(a,b,count=1):
        nonlocal s
        assert s.count(a)==count,(a,s.count(a))
        s=s.replace(a,b)
    edit('from relative_xy import RelativeTarget','from temporal import TemporalTarget, Registration')
    edit('self.relative=RelativeTarget(json.loads', "self.registration=Registration(json.loads(Path(__file__).with_name('view_geometry.json').read_bytes()))\n        self.relative=TemporalTarget(json.loads")
    edit('self.relative=RelativeTarget(self.relative.calibration)','self.relative=TemporalTarget(self.relative.calibration)',2)
    edit("        if self.closed:\n            return base","        if self.closed or self.visual.get('view') is None:\n            return base")
    edit("            mapped=np.r_[uv,1.]@self.aligner.par['matrix']", """            matrix=self.aligner.par['matrix']
            mapped,view=self.registration.locate(self.raw[-1],uv,matrix)
            self.visual.update(view=view,detected_uv=uv.tolist())
            uv=np.linalg.solve(matrix[:2].T,mapped-matrix[2])
            self.relative.archive_view=view==0""")
    edit("            self.visual['aligned']=bool(self.visual['aligned'] and self.relative.centered)", """            edge_centered=self.relative.observe(stamp,self.wrist,query['q'][6],box)
            envelope=self.registration.close_envelopes[view] if view is not None else None
            depth_ok=envelope is not None and tip[2]<=envelope['max_height']
            size_ok=self.relative.visible_width is not None and envelope is not None and self.relative.visible_width>=envelope['min_face_width']
            edge_centered=edge_centered and size_ok
            self.visual['aligned']=bool(depth_ok and distance<=OVERHEAD_TOLERANCE_MM and
                ((self.visual['aligned'] and self.relative.centered) or edge_centered))
            self.visual.update(edge_mode=self.relative.source,edge_centered=edge_centered,depth_ok=depth_ok,size_ok=size_ok)""")
    (root/'ode_edge.py').write_text(s)
    alignment['backend_sha256']=sha(root/'ode_edge.py')
    for n in ('temporal.py','view_geometry.json'):alignment['station_files'][n]=sha(root/n)
    profile['sha256']=hashlib.sha256(json.dumps(alignment,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write('candidate.json',profile)
    write('identity.json',dict(profile=profile['sha256'],backend=sha(root/'ode_edge.py'),geometry=sha(root/'view_geometry.json'),logic=sha(root/'temporal.py'),selection=sha(prev/'old_manifest.json'),base=alignment['base_sha256']))
    print('offline build',profile['sha256'],flush=True)
if __name__=='__main__':build()
