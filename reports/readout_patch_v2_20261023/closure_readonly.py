"""Bind final cache SHA, finished controller and parent receipt; no model load or write."""
import json,hashlib,os,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/READOUT_PATCH_OBJECT41_20261023_v2');O=Q/'output'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((O/'PATCH_MANIFEST.json').read_text());cache=[]
for row in manifest['tasks']:
 p=O/'private'/('patch%d.pth'%row['task']);assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];cache.append(dict(task=row['task'],sha256=sha(p),bytes=p.stat().st_size))
assert (Q/'controller.exit').read_text().strip()=='0'
processes=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  try:env=(p/'environ').read_bytes().split(b'\0')
  except OSError:continue
  if ('PBD_ROOT='+str(Q)).encode() in env:processes.append(int(p.name))
assert not processes
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True);assert not gpu.strip()
print(json.dumps(dict(status='PASS_FINAL_PRIVATE_CACHE_SHA_AND_ZERO_RESIDUAL',cache=cache,own_remaining=processes,gpu=gpu,controller_exit_sha256=sha(Q/'controller.exit'),contract_sha256=sha(Q/'CONTRACT.json'),gate_sha256=sha(O/'A_GATE.json'),patch_manifest_sha256=sha(O/'PATCH_MANIFEST.json'),model_forwards=0,optimizer_calls=0,environment_steps=0,server_writes=0)))
