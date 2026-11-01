"""Repeat observational PMU/clock counts; no benchmark, workload, signals or writes."""
import os,time,json,ctypes,struct,fcntl,subprocess
from pathlib import Path
P=Path('/proc/2169');assert '/S_LAMBDA_R_OBJECT41_42_43_20261101_v1/L005_41/train.py' in (P/'cmdline').read_bytes().decode()
lib=ctypes.CDLL(None,use_errno=True)
def snap():return dict(time=time.time(),stat=(P/'stat').read_text(),schedstat=(P/'schedstat').read_text())
windows=[]
for _ in range(3):
 fds=[];errors={}
 for name,kind,config in [('cycles',0,0),('reference_cycles',0,9),('instructions',0,1),('task_clock',1,1)]:
  buf=ctypes.create_string_buffer(64);struct.pack_into('IIQQQQQIIQ',buf,0,kind,64,config,0,0,3,(1<<0)|(1<<5)|(1<<6),0,0,0)
  fd=lib.syscall(298,ctypes.byref(buf),2169,-1,-1,8)
  if fd<0:errors[name]=ctypes.get_errno()
  else:fds.append((name,fd))
 before=snap()
 for _,fd in fds:fcntl.ioctl(fd,0x2400,0)
 time.sleep(10)
 for _,fd in fds:fcntl.ioctl(fd,0x2401,0)
 after=snap();counts={}
 for name,fd in fds:
  x=struct.unpack('QQQ',os.read(fd,24));os.close(fd);counts[name]=dict(raw=x[0],enabled_ns=x[1],running_ns=x[2])
 windows.append(dict(before=before,after=after,counts=counts,errors=errors))
cpu={}
for i in sorted(os.sched_getaffinity(2169)):
 p=Path('/sys/devices/system/cpu')/('cpu'+str(i))/'cpufreq'
 cpu[str(i)]={n:(p/n).read_text() for n in ['scaling_cur_freq','scaling_max_freq','scaling_governor','cpuinfo_cur_freq'] if (p/n).exists()}
q=subprocess.run(['lscpu'],capture_output=True,text=True)
print(json.dumps(dict(status='READONLY_REPEATED_CPU_COUNTERS_NOT_CALIBRATED_HOST_FREQUENCY',windows=windows,clock_ticks=os.sysconf('SC_CLK_TCK'),cpufreq=cpu,lscpu=q.stdout,notes=['User-only hardware cycles and instructions may be virtualized; do not infer physical CPU MHz without reference-clock validation.','No extra workload or optimizer/environment calls; counts observe the existing main training thread.'],production_writes=0,target_signals=0,extra_optimizer_calls=0,environments=0)))
