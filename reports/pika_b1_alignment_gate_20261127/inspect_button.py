"""Read station sources, process names and archived session tails. No device access."""
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys


def inspect(work):
    station = work / 'piper_test_console_pika_b1_full_v1'
    key = 'B1-ode50-aligned-20261125'
    desktop = (station / 'desktop.py').read_text()
    runner = (station / 'runner.py').read_text()
    methods = {node.name:ast.get_source_segment(runner,node) for node in ast.walk(ast.parse(runner))
               if isinstance(node,ast.FunctionDef) and node.name in ('begin','guard','__init__')}
    profiles = json.loads((station/'profiles.json').read_bytes())
    evidence = json.loads((station/f'verified-{key}.json').read_bytes())
    tz = dt.timezone(dt.timedelta(hours=8))
    clock = lambda t:dt.datetime.fromtimestamp(t,tz).isoformat(timespec='seconds')
    names = ['runner.py','desktop.py','b1_finish.py','grip_window.py','profiles.json']
    result = dict(time=clock(dt.datetime.now().timestamp()), profile=key, bundle=profiles[key]['sha256'],
                  matched_receipt=evidence['sha256']==profiles[key]['sha256'], files={
                      n:hashlib.sha256((station/n).read_bytes()).hexdigest() for n in names},
                  gui_requires_ready="start=self.mode=='preview' and self.ready" in desktop,
                  forced_preview='display.update(ready=False,text=B1_RETURN_PENDING)' in methods['guard'],
                  begin_refuses='if self.aligned_b1:\n            raise ValueError(B1_RETURN_PENDING)' in methods['begin'],
                  pending_text=[ln.split('=',1)[1].strip().strip('"') for ln in runner.splitlines() if ln.startswith('B1_RETURN_PENDING =')],
                  default_return='self.finish = B1Finish()' in runner,
                  processes=[],recent=[])
    for d in Path('/proc').glob('[0-9]*'):
        try:
            raw=[a.decode(errors='replace') for a in (d/'cmdline').read_bytes().split(b'\0') if a]
            if not any(Path(a).name in ('desktop.py','runner.py') for a in raw):continue
            if not(any(str(station) in a for a in raw) or d.joinpath('cwd').resolve()==station):continue
            result['processes'].append([Path(a).name for a in raw if Path(a).name in ('desktop.py','runner.py') or a in ('guard','infer',key)])
        except OSError:pass
    candidates=sorted((work/'policy_runs').glob('*.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True)[:3]
    for path in candidates:
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-300000))
            lines=f.readlines()[1:]
        tail=[]
        for line in lines:
            try: r=json.loads(line)
            except ValueError: continue
            tail.append(r)
        pred=[r for r in tail if r.get('event')=='prediction']
        r=pred[-1] if pred else {}
        result['recent'].append(dict(session=path.stem,mtime=clock(path.stat().st_mtime),
          last_event=tail[-1].get('event') if tail else None,last_prediction_mode=r.get('mode'),
          full=r.get('full'), cameras=sorted((r.get('images') or {}).keys()),feedback_dim=len(r.get('q') or []),
          prediction_dim=len(r.get('p') or []),events_in_tail=sorted({str(x.get('event','')) for x in tail})))
    return result


if __name__=='__main__' and len(sys.argv)>1:
    print(json.dumps(inspect(Path(sys.argv[1])),ensure_ascii=True))
