"""Resolve existing sampled code addresses and read whitelisted configuration only."""
import json,time,subprocess,os,ctypes,struct,bisect,collections
from pathlib import Path
D=Path('/data/mlr_ifa/diagnostics/lambda_native_readonly_20261101_231449')
# Locate the unique prior profile by its request-time prefix, never run another sampler.
dirs=list(Path('/data/mlr_ifa/diagnostics').glob('lambda_native_readonly_20261101_2314*'))
assert len(dirs)==1;D=dirs[0];v=json.loads((D/'RESULT.json').read_text())
P=Path('/proc/2169');assert '/S_LAMBDA_R_OBJECT41_42_43_20261101_v1/L005_41/train.py' in (P/'cmdline').read_bytes().decode()
allowed={'CUDA_LAUNCH_BLOCKING','CUDA_DEVICE_MAX_CONNECTIONS','CUDA_MODULE_LOADING','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NCCL_BLOCKING_WAIT','TORCH_NCCL_BLOCKING_WAIT','CUBLAS_WORKSPACE_CONFIG'}
env={}
for row in (P/'environ').read_bytes().split(b'\0'):
 key,sep,value=row.partition(b'=')
 if key.decode(errors='replace') in allowed:env[key.decode()]=value.decode(errors='replace')
tables={};resolved=[];summary=collections.Counter()
for path in {x['path'] for x in v['addresses']}:
 if not path.startswith('/') or not Path(path).exists():continue
 p=subprocess.run(['nm','-S','-n','--defined-only',path],capture_output=True,text=True,timeout=30);rows=[]
 for line in p.stdout.splitlines():
  s=line.split(None,3)
  if len(s)!=4:continue
  try:a,n=int(s[0],16),int(s[1],16)
  except ValueError:continue
  if n:rows.append((a,n,s[3]))
 tables[path]=sorted(rows)
for x in v['addresses']:
 tab=tables.get(x['path'],[]);addr=int(x['elf_address'],16);i=bisect.bisect_right([z[0] for z in tab],addr)-1
 sym=tab[i][2] if i>=0 and addr<tab[i][0]+tab[i][1] else x['symbol']
 resolved.append(dict(x,full_symbol=sym));summary[(x['path'],sym)]+=x['samples']
fdlist=[];lib=ctypes.CDLL(None,use_errno=True);failures={}
# Counting only, no sample interrupts. Hardware counters can be unavailable in VMs.
for name,kind,config in [('cycles',0,0),('instructions',0,1),('cache_misses',0,3),('context_switches',1,3),('cpu_migrations',1,4)]:
 attr=ctypes.create_string_buffer(64);struct.pack_into('IIQQQQQIIQ',attr,0,kind,64,config,0,0,3,(1<<5)|(1<<6),0,0,0)
 fd=lib.syscall(298,ctypes.byref(attr),2169,-1,-1,8)
 if fd<0:failures[name]=ctypes.get_errno()
 else:fdlist.append((name,fd))
started=time.time();time.sleep(15);counts={}
for name,fd in fdlist:
 values=struct.unpack('QQQ',os.read(fd,24));os.close(fd);counts[name]=dict(raw=values[0],time_enabled_ns=values[1],time_running_ns=values[2])
print(json.dumps(dict(status='READONLY_SYMBOL_AND_COUNTER_FOLLOWUP',source_directory=str(D),whitelisted_initial_environment=env,environment_limit='proc/environ reflects startup environment; runtime explicitly sets OMP/MKL/OpenBLAS and torch threads to one in frozen source.',resolved_top_addresses=resolved,resolved_counts_among_top100=[dict(path=p,symbol=s,samples=n) for (p,s),n in summary.most_common(30)],counter_window_seconds=time.time()-started,counters=counts,counter_unavailable_errno=failures,extra_optimizer_calls=0,environments=0,target_signals=0,production_writes=0)))
