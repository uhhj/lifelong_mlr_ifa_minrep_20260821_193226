from pathlib import Path
R=Path(__file__).resolve().parent
names=['legacy_losses','prototype_loss','losses','synthetic_checks']
src={n:(R/(n+'.py')).read_text(encoding='utf8') for n in names}
code='''import os,sys,types,json,hashlib,time
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import torch
torch.set_num_threads(1)
SOURCES='''+ascii(src)+'''
for n,s in SOURCES.items():
 m=types.ModuleType(n);sys.modules[n]=m;exec(compile(s,n,'exec'),m.__dict__)
tic=time.time();result=sys.modules['synthetic_checks'].run()
print(json.dumps(dict(status='PASS_LOCAL_SOURCE_CPU_ADMISSION',results=result,source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in SOURCES.items()},seconds=time.time()-tic,server_writes=0,cuda_calls=0,observed=time.time())))
'''
(R/'cpu_admission_readonly.py').write_text(code,encoding='utf8')
