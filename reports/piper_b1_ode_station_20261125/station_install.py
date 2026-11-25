"""Publish a verified model profile into the installed PiPER console. No device APIs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

BASE='B1-ode50-20261123'
KEY='B1-ode50-aligned-20261125'
VERSION='11-25 横向纠偏'
STATION='piper_test_console_sdk_v1'
DIAG='offline_diagnostics/b1_grounded_alignment_20261125'
ORIGINAL='f0d72ec94319d9bdc432d93490fc78fffd8f9f8a5bb85f20c30844e5728bc8d2'
ALGORITHM='69aa77c204e3e941ed578b4155749794bffeae43a5353605c8d81f7ccc504b44'
TRACE='2e14ef23b141c697f0ffc201bea42650b02569601d7fadf6f08b5a1204c7aa20'
PARAMETERS={'private_policy.npz':'2445a36f22ed0591b354384217aecceedd1658157a7825b288b525370c41c839',
    'private_heatmap.pt':'3a599ef42eb92c515cfcd8c5ef54a58b38da4e78f1ab1499d4db711a3450ea58',
    'private_fusion.npz':'5df3541cd60ef93c32348b7bc528c3d807f30c542c229cd844169adc5929c318',
    'private_settle.npz':'45f3c1250e053cac92748bf2a529ff2d43ce1b04794886388381b0e3ef137c3d'}
PROTECTED=('desktop.py','catalog.py','bus.py','agilex_arm.py','capture.py','ode_live.py','policy.py','smol_live.py')


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def encode(value):
    return (json.dumps(value,indent=1,ensure_ascii=False,allow_nan=False)+'\n').encode()


def atomic(path,blob):
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        pending=Path(f.name)
        try:
            f.write(blob);f.flush();os.fsync(f.fileno())
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    try:
        os.chmod(pending,path.stat().st_mode&0o777 if path.exists() else 0o644)
        pending.replace(path)
    finally:pending.unlink(missing_ok=True)


def loader(old):
    marker=b"    if profile['family'] == 'ODE':\n        from ode_live import B1Predictor"
    dispatch=b"    if profile['family'] == 'ODE' and 'alignment' in profile:\n        from ode_alignment import GroundedODE\n        return GroundedODE(profile)\n"
    if b'\r\n' in old:
        marker=marker.replace(b'\n',b'\r\n')
        dispatch=dispatch.replace(b'\n',b'\r\n')
    assert old.count(marker)==1 and hashlib.sha256(old).hexdigest()==ORIGINAL
    return old.replace(marker,dispatch+marker,1)


def profile(work,stage):
    station=work/STATION
    parent=read(station/'profiles.json')[BASE]
    assert parent['sha256']==TRACE and sha(work/parent['checkpoint'])==TRACE
    files={}
    algorithm=DIAG+'/grounded.py'
    weights=DIAG+'/causal_appearance'
    assert sha(work/algorithm)==ALGORITHM
    for name,expected in PARAMETERS.items():
        path=weights+'/'+name
        assert sha(work/path)==expected
        files[path]=expected
    for path in [algorithm,weights+'/fit.json',weights+'/inspection.json',parent['adapter'],
                 'offline_diagnostics/b1_feedback_xy_20261125/policy_xy.py',
                 'offline_diagnostics/b1_relative_policy_20261125/experiment.py',
                 'offline_diagnostics/b1_spatial_ablation_20261121_2034/fit_gripper.py']:
        files[path]=sha(work/path)
    bundle=dict(base_sha256=TRACE,backend_sha256=sha(stage/'ode_alignment.py'),algorithm=algorithm,
        weights=weights,files=files,close_mm=55.,radius_mm=60.,ode_inference='zero perturbation')
    ident=hashlib.sha256(json.dumps(bundle,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(parent,version=VERSION,sha256=ident,alignment=bundle)


def stock_runner(work,stage):
    p=work/STATION/'runner.py'
    s=p.read_bytes()
    if hashlib.sha256(s).hexdigest()==ORIGINAL:return s
    old=(stage/'before-runner.py').read_bytes()
    assert s==loader(old)
    return old


def test(work,stage):
    station=work/STATION
    sys.path[:0]=[str(stage),str(station)]
    filename=station/'runner.py'
    mod=types.ModuleType('candidate_runner');mod.__file__=str(filename)
    exec(compile(loader(stock_runner(work,stage)),str(filename),'exec'),mod.__dict__)
    from station_check import replay
    p=profile(work,stage)
    result=replay(work,stage,mod,p,full=True)
    result['protected']={n:sha(station/n) for n in PROTECTED}
    result['loader_sha256']=hashlib.sha256(loader(stock_runner(work,stage))).hexdigest()
    atomic(stage/'offline.json',encode(result))
    print('verified before install',result['pair_checks'],result['timing'],flush=True)


def install(work,stage):
    import fcntl
    station=work/STATION
    p=profile(work,stage)
    checked=read(stage/'offline.json')
    assert checked['profile']==KEY and checked['sha256']==p['sha256'] and checked['accepted']
    lock=(stage/'deployment.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    assert checked['protected']=={n:sha(station/n) for n in PROTECTED}
    original=stock_runner(work,stage)
    updated=loader(original)
    assert hashlib.sha256(updated).hexdigest()==checked['loader_sha256']
    registry=station/'profiles.json'
    before=registry.read_bytes()
    entries=json.loads(before)
    backups=stage/'before-profiles.json'
    if KEY not in entries:
        if backups.exists():assert backups.read_bytes()==before
        else:atomic(backups,before)
    else:assert entries[KEY]==p
    if not (stage/'before-runner.py').exists():atomic(stage/'before-runner.py',original)
    else:assert (stage/'before-runner.py').read_bytes()==original
    model=station/'ode_alignment.py'
    if model.exists():assert model.read_bytes()==(stage/'ode_alignment.py').read_bytes()
    entries[KEY]=p
    atomic(model,(stage/'ode_alignment.py').read_bytes())
    report=station/('verified-'+KEY+'.json')
    atomic(report,encode(checked))
    atomic(station/'runner.py',updated)
    if registry.read_bytes()!=before:raise RuntimeError('策略目录在发布期间改变')
    atomic(registry,encode(entries))
    print('registered',KEY,VERSION,flush=True)


def verify(work,stage):
    station=work/STATION
    sys.path[:0]=[str(station),str(stage)]
    import runner
    from station_check import replay
    catalog=runner.PROFILES
    prior=read(stage/'before-profiles.json')
    assert {k:v for k,v in catalog.items() if k!=KEY}==prior
    assert catalog[KEY]==profile(work,stage)
    proof=read(station/('verified-'+KEY+'.json'))
    assert proof['accepted'] and proof['protected']=={n:sha(station/n) for n in PROTECTED}
    assert proof['loader_sha256']==sha(station/'runner.py')
    actual=replay(work,stage,runner,catalog[KEY],full=False)
    import ode_alignment
    assert Path(ode_alignment.__file__).resolve()==station/'ode_alignment.py'
    assert actual['windows']==proof['windows'][:len(actual['windows'])] and len(actual['windows'])==3
    desktop=Path(subprocess.check_output(['xdg-user-dir','DESKTOP'],text=True).strip())/'PiPER-Policy.desktop'
    assert 'PiPER 通用策略测试台' in desktop.read_text() and str(station/'desktop.py') in desktop.read_text()
    ui='desktop shortcut and registered selector'
    if shutil.which('xvfb-run') and shutil.which('xauth'):
        probe="""from desktop import Console
import tkinter as tk,sys
root=tk.Tk()
root.withdraw()
app=Console(root)
app.task.current(next(i for i,v in enumerate(app.task['values']) if v.split()[0]=='B1'))
app.choose_task()
app.model.set('ODE')
app.choose_model()
app.version.set(sys.argv[2])
app.selected()
assert app.key==sys.argv[1] and app.verified and app.proc is None
assert str(app.buttons['preview']['state'])=='normal' and str(app.buttons['start']['state'])=='disabled'
root.destroy()
"""
        subprocess.run(['xvfb-run','-a','/usr/bin/python3','-B','-c',probe,KEY,VERSION],cwd=station,check=True,timeout=30)
        ui='existing Tk window selected on isolated virtual display'
    receipt=dict(profile=KEY,version=VERSION,bundle_sha256=catalog[KEY]['sha256'],base_sha256=TRACE,
        original_profiles=len(prior),station_profiles=len(catalog),profile_registry_sha256=sha(station/'profiles.json'),
        runner_sha256=sha(station/'runner.py'),protected=proof['protected'],window_count=len(actual['windows']),
        installed_backend_module=Path(ode_alignment.__file__).name,entrypoint='PiPER-Policy.desktop',
        selection_check=ui,shortcut_sha256=sha(desktop),working_mode='archive replay only')
    atomic(stage/'installed.json',encode(receipt))
    print('installed and reloaded',receipt,flush=True)


if __name__=='__main__':
    work,stage=map(Path,sys.argv[1:3]);cmd=sys.argv[3]
    if cmd=='test':test(work,stage)
    elif cmd=='install':install(work,stage)
    elif cmd=='verify':verify(work,stage)
    elif cmd=='inspect':print(profile(work,stage)['sha256'],flush=True)
