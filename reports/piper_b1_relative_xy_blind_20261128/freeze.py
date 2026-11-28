"""Build an offline backend from the frozen November 27 station. Never install or start a device."""
from pathlib import Path
import sys,json,hashlib

root=Path(__file__).resolve().parent
work,old=map(Path,sys.argv[1:3])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
profile=json.loads((old/'profiles.json').read_bytes())['B1-ode50-aligned-20261125']
assert profile['sha256']=='27fca3b2b3252827076d08aff0a122e3ec12004e07959ce7992c0b74e0b3d88d'
bundle=profile['alignment']
assert sha(old/'ode_alignment.py')==bundle['backend_sha256']
source=(old/'ode_alignment.py').read_text()
source=source.replace('from wrist_cue import ConfirmedCue, modified_step, signed_cue','from relative_xy import RelativeTarget')
source=source.replace('self.cue=ConfirmedCue()', "self.relative=RelativeTarget(json.loads(Path(__file__).with_name('relative_camera.json').read_bytes()))",1)
i=source.index('        if self.mode is None:');j=source.index('        if not np.isfinite(corrected)',i)
source=source[:i]+'''        start=aligner.fk(q[-1])
        coarse=np.r_[uv,1.]@aligner.par['matrix']
        registered=self.algorithm.settle_goal(uv,q,self.wrist,aligner.par,self.unguarded,aligner.flange)
        step=self.relative.step(start,coarse,registered,float(aligner.par['gain']),float(aligner.par['cap']))
        path=start+np.arange(1,9)[:,None]/8*step
        corrected,errors=aligner.xy.joint_commands(base,path,aligner.par['tip'])
        if max(errors[k] for k in ('xy_max_mm','height_max_mm','angle_max_deg'))>1e-4:
            raise ValueError('Relative B1 XY reconstruction exceeded tolerance')
'''+source[j:]
source=source.replace('self.radius=bundle', "self.unguarded={k:v for k,v in self.aligner.steady.items() if k!='guard'}\n        self.radius=bundle")
source=source.replace('            self.cue=ConfirmedCue()', '            self.relative=RelativeTarget(self.relative.calibration)')
source=source.replace('            self.heat.clear()','            self.relative=RelativeTarget(self.relative.calibration)\n            self.heat.clear()')
i=source.index("            z0,ramp");j=source.index('        small=ode_pixels',i)
source=source[:i]+'''            self.relative.update(stamp,self.wrist,tip,self.pose[:3,:3],mapped,query['q'][6])
            self.visual['aligned']=bool(self.visual['aligned'] and self.relative.centered)
            self.visual['fresh_metric_center']=self.relative.centered
        self.mode=None
        self.visual.update(wrist_side=None,confirmed_side=None)
'''+source[j:]
(root/'ode_relative.py').write_text(source)
(root/'relative_camera.json').write_text(json.dumps(json.loads((old/'camera_models.json').read_bytes())['train']['calibration']))
# The backend validates all inherited dependencies relative to itself.
import shutil,copy
for name,h in bundle['station_files'].items():
    assert sha(old/name)==h
    shutil.copyfile(old/name,root/name)
candidate=copy.deepcopy(profile)
c=candidate['alignment'];c['backend_sha256']=sha(root/'ode_relative.py')
for name in ('relative_xy.py','relative_camera.json'):
    c['station_files'][name]=sha(root/name)
candidate['sha256']=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(root/'candidate.json').write_text(json.dumps(candidate))
source=work/'batches/B1_20261123'
blind=sorted(x.parent.name for x in (source/'failed').glob('*/alignment.jsonl'))
rank=json.loads((work/'offline_diagnostics/b1_ode_lerobot_20261123/selection.json').read_bytes())
assert len(blind)==7 and not set(blind)&{r['id'] for r in rank['ranking']}
frozen=dict(baseline=profile['sha256'],candidate=candidate['sha256'],protocol_sha256=sha(root/'protocol.md'),blind=blind,
            backend_sha256=c['backend_sha256'],relative_sha256=c['station_files']['relative_xy.py'],
            camera_sha256=c['station_files']['relative_camera.json'])
dest=root/'frozen.json'
if dest.exists(): assert json.loads(dest.read_bytes())==frozen
else: dest.write_text(json.dumps(frozen,indent=2))
print('frozen',candidate['sha256'],len(blind))
