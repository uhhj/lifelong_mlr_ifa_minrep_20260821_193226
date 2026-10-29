import json,time,hashlib
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1');E=Q/'engineering_copy_20261029_v1'
out=dict(time=time.time(),engineering={},training={},processes=[])
for p in E.glob('*.json'):
 out['engineering'][p.name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),data=json.loads(p.read_text()))
for seed in [41,42,43]:
 r=Q/('seed%d'%seed);h=json.loads((r/'EPOCHS_S.json').read_text()) if (r/'EPOCHS_S.json').exists() else []
 out['training'][str(seed)]=dict(epochs=len(h),last=h[-1] if h else None,train_exit=(r/'train_S.exit').read_text() if (r/'train_S.exit').exists() else None)
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();st=(p/'stat').read_text().split(') ',1)[1].split()
  if str(Q) in cmd or p.name in ['3013','5055','5060']:
   out['processes'].append(dict(pid=int(p.name),state=st[0],start=st[19],exit_code=int(st[49]),cmd=cmd))
 except OSError:pass
out['validation_log_tail']=(E/'validation.log').read_text()[-7000:] if (E/'validation.log').exists() else None
out['controller_log_tail']=(E/'controller.log').read_text()[-4000:] if (E/'controller.log').exists() else None
print(json.dumps(out))
