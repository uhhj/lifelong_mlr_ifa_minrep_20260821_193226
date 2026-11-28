"""Create a content-hashed offline profile from the previous frozen candidate."""
from basis import *
import shutil
def build():
    profile=read(edge/'candidate.json')
    assert profile['sha256']=='793bb192589dd9e5c935808dd8e168c6cc1ad43d4d271b7b6ee5d0861cc5647a'
    bundle=profile['alignment']
    assert rp.sha(edge/'ode_edge.py')==bundle['backend_sha256']
    for name,h in bundle['station_files'].items():
        assert rp.sha(edge/name)==h
        shutil.copyfile(edge/name,dest/name)
    s=(edge/'ode_edge.py').read_text()
    def replace(a,b,n=1):
        nonlocal s
        assert s.count(a)==n,(a,s.count(a));s=s.replace(a,b)
    replace('from temporal import TemporalTarget, Registration','from temporal import Registration\nfrom wrist_lines import WristEvidence\nfrom overhead_track import StationaryCube')
    replace('self.relative=TemporalTarget(', 'self.relative=WristEvidence(',3)
    replace("        self.registration=Registration", "        self.widths=json.loads(Path(__file__).with_name('widths.json').read_bytes())\n        self.registration=Registration")
    replace("            self.relative=WristEvidence(self.relative.calibration)", "            self.tracker=StationaryCube()\n            self.relative=WristEvidence(self.relative.calibration)",2)
    replace('        self.wrist=cv2.cvtColor(pixels[1]',"""        single_uv=self.aligner.det.positions(self.heat[-1])[0].cpu().numpy()
        matrix=self.aligner.par['matrix']
        _,view=self.registration.locate(self.raw[-1],single_uv,matrix)
        tracked,track_mode=self.tracker.update(stamp,self.raw[-1],single_uv,view)
        self.wrist=cv2.cvtColor(pixels[1]""")
    replace("            mapped,view=self.registration.locate(self.raw[-1],uv,matrix)", """            raw=uv.copy()
            mapped=(self.aligner.fk(query['q']) if tracked is None else self.registration.locate(self.raw[-1],tracked,matrix)[0])
            self.visual.update(tracked=tracked is not None,track_mode=track_mode,
                tracker_uv=None if tracked is None else tracked.tolist(),raw_gap=None if tracked is None else float(np.linalg.norm(tracked-raw)))""")
    replace("        if self.closed or self.visual.get('view') is None:\n            return base","""        if self.closed:
            return base
        hold=self.visual.get('track_mode')=='held' and self.visual.get('raw_gap',0)>25 and not self.guiding and self.visual['distance_mm']>self.radius
        if not self.visual.get('tracked') or hold:
            anchor_xy=self.aligner.fk(q[-1])
            corrected,errors=self.aligner.xy.joint_commands(base,np.repeat(anchor_xy[None,:],8,axis=0),self.aligner.par['tip'])
            if max(errors[k] for k in ('xy_max_mm','height_max_mm','angle_max_deg'))>1e-4 or not np.array_equal(corrected[:,6],base[:,6]):
                raise ValueError('Unobserved XY hold failed reconstruction')
            self.details.update(phase='hold_unobserved_xy')
            return corrected""")
    replace("            edge_centered=edge_centered and size_ok", """            if self.relative.source=='three_visible_sides' and view is not None:
                size_ok=self.relative.visible_width>=self.widths[view]['min_width']
            edge_centered=edge_centered and size_ok""")
    replace("self.visual['aligned']=bool(depth_ok and distance", "self.visual['aligned']=bool(tracked is not None and depth_ok and distance")
    replace('from overhead_track import StationaryCube','from overhead_track import StationaryCube\nfrom wrist_xy import WristPosition')
    replace('        self.widths=json.loads', "        self.wrist_fits=json.loads(Path(__file__).with_name('private_wrist_fit.json').read_bytes())\n        self.widths=json.loads")
    replace('            self.tracker=StationaryCube()', '            self.tracker=StationaryCube()\n            self.wrist_position=WristPosition(self.wrist_fits)',2)
    replace("        self.visual={'stamp':stamp", """        tf=self.aligner.flange(np.rad2deg(query['q'][:6]))
        tip_now=tf[:3,3]+tf[:3,:3]@self.aligner.par['tip']
        wrist_xy=self.wrist_position.update(stamp,view,self.wrist,tip_now,tf[:3,:3],self.relative.calibration)
        self.visual={'stamp':stamp""")
    replace("            self.visual.update(tracked=tracked is not None,track_mode=track_mode,", """            raw_xy,_=self.registration.locate(self.raw[-1],raw,matrix)
            usable=wrist_xy is not None and (tracked is None or np.linalg.norm(mapped-wrist_xy)<=26)
            corroborated=usable and np.linalg.norm(raw_xy-wrist_xy)<=12 and np.linalg.norm(raw_xy-tip_now[:2])<=20
            self.relative.wrist_goal=wrist_xy if usable and track_mode!='visible' else None
            self.relative.uncertainty=self.wrist_fits[view]['uncertainty'] if view is not None else 0
            if tracked is None and usable:mapped=wrist_xy
            self.visual.update(guide=usable or tracked is not None,close_coarse=tracked is not None or corroborated,
                wrist_guide=self.relative.wrist_goal is not None,effective_xy=mapped.tolist(),
                tracked=tracked is not None,track_mode=track_mode,""")
    replace("if not self.visual.get('tracked') or hold:", "if not self.visual.get('guide') or (hold and not self.visual.get('wrist_guide')):")
    replace("bool(tracked is not None and depth_ok and distance", "bool(self.visual['close_coarse'] and depth_ok and distance")
    (dest/'ode_tracked.py').write_text(s)
    bundle['backend_sha256']=rp.sha(dest/'ode_tracked.py')
    for n in ('wrist_lines.py','overhead_track.py','widths.json','wrist_xy.py','private_wrist_fit.json'):bundle['station_files'][n]=rp.sha(dest/n)
    profile['sha256']=hashlib.sha256(json.dumps(bundle,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dump('candidate.json',profile)
    dump('identity.json',dict(profile=profile['sha256'],baseline=read(edge/'identity.json')['profile'],backend=bundle['backend_sha256'],new_files={n:rp.sha(dest/n) for n in ('wrist_lines.py','overhead_track.py','widths.json','wrist_xy.py','private_wrist_fit.json')}))
    print('candidate',profile['sha256'],flush=True)
if __name__=='__main__':build()
