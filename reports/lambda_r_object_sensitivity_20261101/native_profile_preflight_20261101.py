import json,os,sys,ctypes,struct,platform,shutil,subprocess,time,re
from pathlib import Path
p=Path('/proc/2169');cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert '/S_LAMBDA_R_OBJECT41_42_43_20261101_v1/L005_41/train.py' in cmdline
out=dict(observed=time.time(),machine=platform.machine(),kernel=platform.release(),tools={x:shutil.which(x) for x in ['readelf','nm','objdump','addr2line','perf']},training_cmd=cmdline)
header=Path('/usr/include/linux/perf_event.h')
if header.exists():out['perf_header_relevant']='\n'.join(x for x in header.read_text().splitlines() if any(k in x for k in ['PERF_ATTR_SIZE_VER0','PERF_TYPE_SOFTWARE','PERF_COUNT_SW_CPU_CLOCK','PERF_SAMPLE_IP','PERF_SAMPLE_TID','PERF_SAMPLE_TIME','PERF_RECORD_SAMPLE','PERF_RECORD_LOST','PERF_EVENT_IOC_ENABLE','PERF_EVENT_IOC_DISABLE','PERF_EVENT_IOC_RESET']))
assert platform.machine()=='x86_64'
a=ctypes.create_string_buffer(64);struct.pack_into('IIQQQQQIIQ',a,0,1,64,0,0,0,0,(1<<0)|(1<<5)|(1<<6),0,0,0)
lib=ctypes.CDLL(None,use_errno=True);fd=lib.syscall(298,ctypes.byref(a),2169,-1,-1,8);out['perf_probe']=dict(fd_valid=fd>=0,errno=ctypes.get_errno() if fd<0 else 0,enabled=False,signals=0)
if fd>=0:os.close(fd)
q=subprocess.run(['dmesg','--level=err,warn'],capture_output=True,text=True);out['nvidia_kernel_errors']=dict(exit=q.returncode,lines=[x for x in q.stdout.splitlines() if re.search(r'NVRM|Xid|nvidia|PCI.*error|AER:',x,re.I)][-30:],stderr=q.stderr[-300:])
q=subprocess.run(['nvidia-smi','--query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max','--format=csv,noheader'],capture_output=True,text=True);out['pcie']=dict(exit=q.returncode,stdout=q.stdout,stderr=q.stderr)
out.update(extra_optimizer_calls=0,environments=0,production_writes=0,signals=0)
print(json.dumps(out))
