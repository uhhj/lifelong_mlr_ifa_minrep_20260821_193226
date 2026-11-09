import os,sys,json,io,time,hashlib,traceback
from pathlib import Path
ENG=Path(__file__).resolve().parent
ROOT=ENG.parent.parent/'S_Q1_OBJECT41_42_43_20261109_rerun_v1'
case=sys.argv[1];assert case in ['Q1_41','Q1_42']
sys.argv=[str(ROOT/'train_wrapper.py'),case]
source=(ROOT/'train_wrapper.py').read_text();assert source.endswith('tr.main()\n')
ns={'__file__':str(ROOT/'train_wrapper.py'),'__name__':'resume_wrapper'}
exec(compile(source[:-len('tr.main()\n')],str(ROOT/'train_wrapper.py'),'exec'),ns)
rt=ns['rt'];tr=ns['tr'];torch=ns['torch']
binding=json.loads((ENG/'RECOVERY_BINDING.json').read_text())['cases'][case]

D=ENG/'validation'/case;D.mkdir(parents=True,exist_ok=False)
def write(p,o):
 p=D/Path(p).name;t=p.with_suffix('.tmp');t.write_text(json.dumps(o,indent=2));os.replace(t,p)
rt.write=write;tr.write=write
ledger=dict(status='RUNNING',optimizer_started=0,optimizer_completed=0,environment_calls=0,production_epochs=0)
def record():write('STATUS.json',ledger)
record()
original=tr.update
def counted(*a,**k):
 ledger['optimizer_started']+=1;record()
 out=original(*a,**k);ledger['optimizer_completed']+=1;return out
tr.update=counted
tr.dist.init_process_group('nccl');torch.cuda.set_device(0)
try:
 rt.verify();cp_path=Path(binding['checkpoint']);assert rt.sha(cp_path)==binding['checkpoint_sha256']
 cp=torch.load(cp_path,map_location='cpu');a=rt.make('S');ds=rt.Dataset(9);loader,sampler=rt.setup(a,ds);info=rt.js(rt.O/'ra_coreset/ifa_task9.json')['info']
 memory=tr.load_memory('S',9);scale=tr.scale_for(ds);assert scale.tolist()==rt.js(rt.R/'TRAIN_INPUT_S_9.json')['scale']
 pr=tr.setup_probes(ds,memory);assert json.loads(json.dumps(pr['metadata']))==rt.js(rt.R/'PROBES_9.json')
 rt.restore(a,cp['state']);assert rt.equal(cp['state'],rt.capture(a))
 stream=io.BytesIO();torch.save(rt.capture(a),stream);stream.seek(0);state=torch.load(stream,map_location='cpu');rt.restore(a,state);assert rt.equal(state,rt.capture(a))
 ledger.update(full_state_restore_exact=True,serialized_restore_exact=True,scale_and_probes_exact=True);record()
 sampler.set_epoch(cp['next_epoch']);order=list(iter(sampler));a.policy.train();iterator=iter(loader)
 for i,expected in enumerate(binding['partial_gates']):
  block=[next(iterator) for _ in range(expected['steps'])]
  base=ns['branch_capture'](a)
  cr=[tr.update(a,b,'C',memory,scale,info,9,retain_weight=0.) for b in block];cs=ns['branch_capture'](a);cv=tr.probe(a,pr,memory)
  rt.restore(a,base)
  qr=[tr.update(a,b,'Q',memory,scale,info,9) for b in block];qs=ns['branch_capture'](a);qv=tr.probe(a,pr,memory)
  decision,checks=tr.choose(cv,qv);selected=cs if decision=='C' else qs;rt.restore(a,selected);assert rt.equal(selected,ns['branch_capture'](a))
  gate=dict(stage=9,epoch=cp['next_epoch'],batch=i*10,steps=len(block),start_step=base['step'],end_step=a.step,C=cv,Q=qv,accepted=decision,checks=checks)
  assert gate==expected,('Replay differs',case,i,gate,expected)
  ledger['exact_gate_blocks']=i+1;record()
 assert ledger['optimizer_started']==ledger['optimizer_completed']==binding['recorded_partial_physical']
 assert rt.sha(cp_path)==binding['checkpoint_sha256']
 assert all(rt.sha(rt.R/n)==h for n,h in binding['light_hashes'].items())
 rt.verify();ledger.update(status='PASS_FULL_RESTORE_AND_ORIGINAL_PARTIAL_GATES_EXACT',finished=time.time(),original_production_files_unchanged=True);record()
except BaseException:
 ledger.update(status='FAILED_PRESERVED_NO_PRODUCTION_RESUME',error=traceback.format_exc());record();raise
finally:tr.dist.destroy_process_group()
