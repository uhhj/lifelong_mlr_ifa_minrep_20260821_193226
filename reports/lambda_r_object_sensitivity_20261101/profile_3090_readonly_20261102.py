"""60s low-rate kernel IP + nonblocking Python sampling; no target signals/writes."""
import os,sys,json,time,struct,ctypes,mmap,fcntl,subprocess,hashlib,collections,bisect
from pathlib import Path
P=Path('/proc/2566');Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
assert str(Q/'engineering_resume_3090_20261102_v1/train_wrapper.py') in (P/'cmdline').read_bytes().replace(b'\0',b' ').decode()
tool=Path('/data/mlr_ifa/diagnostics/lambda_r_profile_20261101_v1/py-spy-0.4.2')
assert hashlib.sha256(tool.read_bytes()).hexdigest()=='9b4d1f39b2a47ae44f4c6a46f615dcc0287d7755beba5065f32391951e07d594'
D=Path('/data/mlr_ifa/diagnostics')/('lambda_3090_native_readonly_'+time.strftime('%Y%m%d_%H%M%S'));D.mkdir()
def snap():
 return dict(time=time.time(),stat=(P/'stat').read_text(),schedstat=(P/'schedstat').read_text(),io=(P/'io').read_text(),status=json.loads((Q/'L005_41/TRAIN_STATUS.json').read_text()))
def command(args,timeout=20):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout);return dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
maps=(P/'maps').read_text();regions=[]
for line in maps.splitlines():
 a=line.split(None,5)
 if len(a)<6 or not a[5].startswith('/'):continue
 lo,hi=[int(x,16) for x in a[0].split('-')];regions.append((lo,hi,int(a[2],16),a[5]))
attr=ctypes.create_string_buffer(64)
# PERF_TYPE_SOFTWARE/CPU_CLOCK; 99Hz; sample IP,TID,TIME; exclude kernel/hypervisor.
struct.pack_into('IIQQQQQIIQ',attr,0,1,64,0,99,7,0,(1<<0)|(1<<5)|(1<<6)|(1<<10),1,0,0)
lib=ctypes.CDLL(None,use_errno=True);fd=lib.syscall(298,ctypes.byref(attr),2566,-1,-1,8)
assert fd>=0,ctypes.get_errno()
page=os.sysconf('SC_PAGESIZE');ring=mmap.mmap(fd,page*65,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ|mmap.PROT_WRITE)
offset=struct.unpack_from('Q',ring,1040)[0] or page;size=struct.unpack_from('Q',ring,1048)[0] or page*64
tail=0;samples=[];lost=0;gpu=[]
def readring(pos,n):
 pos%=size;k=min(n,size-pos);return ring[offset+pos:offset+pos+k]+(ring[offset:offset+n-k] if n>k else b'')
def drain():
 global tail,lost
 head=struct.unpack_from('Q',ring,1024)[0]
 assert head-tail<=size,'Ring overflow; do not silently lose counts'
 while tail<head:
  typ,misc,n=struct.unpack('IHH',readring(tail,8));assert n>=8 and tail+n<=head
  data=readring(tail,n)
  if typ==9:
   ip,pid,tid,tm=struct.unpack_from('QIIQ',data,8);samples.append((ip,pid,tid,tm))
  elif typ==2:lost+=struct.unpack_from('QQ',data,8)[1]
  tail+=n
 struct.pack_into('Q',ring,1032,tail)
before=snap();spyfile=D/'python_stacks.txt'
spy=subprocess.Popen([str(tool),'record','--pid','2566','--nonblocking','--idle','--threads','--full-filenames','--rate','10','--duration','60','--format','raw','--output',str(spyfile)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
try:
 fcntl.ioctl(fd,0x2403,0);fcntl.ioctl(fd,0x2400,0);start=time.monotonic();next_gpu=0
 while time.monotonic()-start<60:
  elapsed=time.monotonic()-start
  if elapsed>=next_gpu:
   gpu.append(dict(time=time.time(),result=command(['nvidia-smi','--query-gpu=utilization.gpu,power.draw,temperature.gpu,clocks.current.sm','--format=csv,noheader'])));next_gpu+=5
  drain();time.sleep(.1)
 fcntl.ioctl(fd,0x2401,0);drain();cpu_clock_ns=struct.unpack('Q',os.read(fd,8))[0]
finally:
 ring.close();os.close(fd)
stdout,stderr=spy.communicate(timeout=20);after=snap()
counts=collections.Counter();addresses=collections.Counter()
for ip,pid,tid,tm in samples:
 found=next((z for z in regions if z[0]<=ip<z[1]),None)
 path=found[3] if found else '[unknown]';foff=ip-found[0]+found[2] if found else ip
 counts[path]+=1;addresses[(path,foff)]+=1
resolved=[];symbol_tables={};loads={};errors={}
for path,num in counts.most_common():
 if not path.startswith('/') or not Path(path).exists():continue
 try:
  headers=command(['readelf','-lW',path]);segments=[]
  for line in headers['stdout'].splitlines():
   parts=line.split()
   if parts and parts[0]=='LOAD':segments.append(tuple(int(parts[i],16) for i in [1,2,4]))
  loads[path]=segments
  syms=command(['nm','-D','-S','-n','--defined-only',path],timeout=30);table=[]
  for line in syms['stdout'].splitlines():
   parts=line.split(None,3)
   if len(parts)==4:
    try:addr=int(parts[0],16);length=int(parts[1],16)
    except ValueError:continue
    if length:table.append((addr,length,parts[3]))
  symbol_tables[path]=sorted(table)
 except Exception as exc:errors[path]=str(exc)
funcs=collections.Counter()
for (path,foff),n in addresses.items():
 seg=next((x for x in loads.get(path,[]) if x[0]<=foff<x[0]+x[2]),None);va=foff-seg[0]+seg[1] if seg else foff
 table=symbol_tables.get(path,[]);i=bisect.bisect_right([x[0] for x in table],va)-1
 symbol=table[i][2] if i>=0 and va<table[i][0]+table[i][1] else '[unresolved]'
 funcs[(path,symbol)]+=n
 resolved.append(dict(path=path,file_offset=hex(foff),elf_address=hex(va),samples=n,symbol=symbol))
resolved.sort(key=lambda x:x['samples'],reverse=True)
disassembly=[]
for x in resolved[:5]:
 if x['path'].startswith('/') and Path(x['path']).exists():
  addr=int(x['elf_address'],16);z=command(['objdump','-d','--start-address='+str(max(0,addr-48)),'--stop-address='+str(addr+80),x['path']]);disassembly.append(dict(address=x,output=z))
out=dict(status='READONLY_NATIVE_IP_AND_NONBLOCKING_PYTHON_PROFILE',directory=str(D),before=before,after=after,cpu_clock_ns=cpu_clock_ns,sampling_frequency_hz=99,sample_count=len(samples),lost_samples=lost,modules=[dict(path=p,samples=n) for p,n in counts.most_common()],functions=[dict(path=p,symbol=s,samples=n) for (p,s),n in funcs.most_common(30)],addresses=resolved[:100],disassembly=disassembly,symbol_errors=errors,gpu_samples=gpu,python_profile=dict(exit=spy.returncode,stdout=stdout,stderr=stderr,stacks=spyfile.read_text() if spyfile.exists() else None),extra_optimizer_calls=0,environments=0,production_writes=0,target_signals=0,limits=['Kernel CPU-clock IP samples measure active main-thread user instructions, not GPU kernel duration.','No ptrace stop, no injected code, no stack locals or model data.','Symbol resolution only names symbols whose declared address ranges contain the sample; others remain unresolved.','Concurrent low-rate sampling may add small measurement overhead; no claim of zero wall-clock effect.'])
(D/'RESULT.json').write_text(json.dumps(out));print(json.dumps(out))
