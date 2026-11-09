import contextlib,io,json,hashlib
stream=io.StringIO()
with contextlib.redirect_stdout(stream):
 exec(compile('"""Read-only observation of the new full-S root, including split gate records."""\nimport json,os,hashlib,time,subprocess\nfrom pathlib import Path\nQ=Path(\'/data/mlr_ifa/experiment_workspaces/S_Q1_OBJECT41_42_43_20261109_rerun_v1\')\nPRIVATE=Path(\'/data/mlr_ifa/experiments\')/Q.name\nOLD=Path(\'/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FORKS_20261027_v1\')\nsha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()\nc=json.loads((Q/\'CONTRACT.json\').read_text());assert all(sha(p)==h for p,h in c[\'files\'].items())\nfor case in [\'Q1_41\', \'Q1_42\', \'Q1_43\']:\n q=Q/case;cc=json.loads((q/\'CONTRACT.json\').read_text());assert sha(q/\'CONTRACT.json\')==c[\'children\'][case]\n assert all(sha(Path(cc[\'source_root\'])/n)==h for n,h in cc[\'parent_source\'].items())\nprocs=[];oldprocs=[]\nfor p in Path(\'/proc\').glob(\'[0-9]*\'):\n try:\n  cmd=(p/\'cmdline\').read_bytes().replace(b\'\\0\',b\' \').decode(errors=\'replace\')\n  if int(p.name)!=os.getpid():\n   item=dict(pid=int(p.name),cmd=cmd,state=(p/\'stat\').read_text().split(\')\')[-1].split()[0])\n   if str(Q) in cmd:procs.append(item)\n   if str(OLD) in cmd:oldprocs.append(item)\n except OSError:pass\nassert not oldprocs,\'Canceled mechanism must remain stopped\'\nlight={};gate_sources={}\ndef add(name,raw):\n light[name]=dict(text=raw.decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))\nfor p in Q.rglob(\'*\'):\n if not p.is_file() or p.suffix not in [\'.json\',\'.jsonl\',\'.exit\'] or p.name in [\'CONTRACT.json\',\'PARENT_BINDING.json\',\'PREDEPLOY.json\']:continue\n if p.name==\'GATES.jsonl\':\n  raw=p.read_bytes();full=raw[:raw.rfind(b\'\\n\')+1];parts={}\n  for line in full.splitlines(True):\n   g=json.loads(line);parts.setdefault(g[\'stage\'],[]).append(line)\n  for stage,lines in parts.items():add(p.parent.name+\'/GATES_stage%d.jsonl\'%stage,b\'\'.join(lines))\n  gate_sources[str(p.relative_to(Q))]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),complete_bytes=len(full),partial_line_bytes=len(raw)-len(full))\n else:\n  assert p.stat().st_size<8*1024**2,str(p)\n  add(str(p.relative_to(Q)),p.read_bytes())\ncapacity={};allocated=0;seen=set()\nfor p in (p for b in map(Path,[\'/data/mlr_ifa/experiments/S_Q1_OBJECT41_42_43_20261109_rerun_v1\', \'/data/mlr_ifa/experiments/S_Q1_GOAL41_42_43_20261109_rerun_v1\']) for p in (b.rglob(\'*\') if b.exists() else [])):\n if not p.is_file() or p.is_symlink():continue\n st=p.stat();key=(st.st_dev,st.st_ino)\n if key not in seen:seen.add(key);allocated+=st.st_blocks*512\nfor mount,bound,reserve,usefree in [(\'/root\',128*1024**2,1024**3,False),(\'/data\',6*1024**3,2*1024**3,True)]:\n v=os.statvfs(mount);free=(v.f_bfree if usefree else v.f_bavail)*v.f_frsize;remaining=max(0,bound-allocated) if mount==\'/data\' else bound\n capacity[mount]=dict(free=free,remaining_bound=remaining,reserve=reserve,spare=free-remaining-reserve)\ngpu=subprocess.run([\'nvidia-smi\',\'--query-compute-apps=pid\',\'--format=csv,noheader\'],capture_output=True,text=True)\nprint(json.dumps(dict(status=\'READ_ONLY_SNAPSHOT_NOT_FINAL\',observed=time.time(),contract_sha256=sha(Q/\'CONTRACT.json\'),children=c[\'children\'],source_files=len(c[\'files\']),processes=procs,canceled_mechanism_processes=oldprocs,gpu_diagnostic=dict(exit=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr),capacity=capacity,light=light,gate_sources=gate_sources,log_tails={str(p.relative_to(Q)):p.read_text(errors=\'replace\')[-4000:] for p in Q.rglob(\'*.log\')})))\n',"OBJECT/check_readonly.py","exec"))
data=json.loads(stream.getvalue())
M=Path('/data/mlr_ifa/experiment_workspaces/S_Q1_OBJECT_GOAL41_42_43_20261109_rerun_v1')
mc=json.loads((M/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in mc['files'].items())
data['coordinator']={n:(M/n).read_text() for n in ['STATUS.json','EVENTS.jsonl','controller.exit','FAILURE.json'] if (M/n).exists()}
data['coordinator_contract_sha256']=sha(M/'CONTRACT.json')

ENG=Path('/data/mlr_ifa/experiment_workspaces/S_Q1_OBJECT_GOAL41_42_43_20261109_rerun_v1/engineering_reboot_resume_20261109_v1')

ec=json.loads((ENG/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in ec['files'].items())
data['engineering_contract_sha256']=sha(ENG/'CONTRACT.json')
data['engineering']={str(p.relative_to(ENG)):dict(text=p.read_text(),sha256=sha(p)) for p in ENG.glob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.exit']}
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid() and str(ENG) in cmd:data['processes'].append(dict(pid=int(p.name),cmd=cmd,state=(p/'stat').read_text().split(') ',1)[1].split()[0]))
 except OSError:pass
data['processes']=list({x['pid']:x for x in data['processes']}.values())
prefix={}
for case,b in json.loads((ENG/'RECOVERY_BINDING.json').read_text())['cases'].items():
 r=Q/case;old=ENG/'original_light'/case
 h=json.loads((r/'EPOCHS_S.json').read_text());original=json.loads((old/'EPOCHS_S.json').read_text());assert h[:len(original)]==original
 gates=(r/'GATES.jsonl').read_bytes();oldg=(old/'GATES.jsonl').read_bytes();assert gates.startswith(oldg)
 prefix[case]=dict(complete_epoch_prefix=len(original),original_gate_rows=b['gate_rows_total'],exact=True,current_epoch_count=len(h))
data['recovery_prefix']=prefix
data['engineering_log_tails']={p.name:p.read_text(errors='replace')[-2500:] for p in ENG.glob('*.log')}
print(json.dumps(data))
