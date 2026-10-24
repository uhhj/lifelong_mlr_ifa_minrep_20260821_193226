# Appended to external native CPU policy loader by build_final_wrapper.py.
contract=js(Q/'CONTRACT.json');assert sha(contract['official_initialization']['path'])==contract['official_initialization']['sha256']
if (Q/'REPAIR_CONTINUATION.json').exists():
 binding=js(Q/'REPAIR_CONTINUATION.json');restored=js(Q/'RESTORE_ADMISSION.json');assert restored['status']=='PASS_ACTUAL_GPU_FULL_STATE_RESTORE' and restored['all_fields_exact'] and restored['completed_epochs_repeated']==0 and restored['optimizer_updates']==restored['forwards']==restored['environments']==0
 assert restored['recovery_sha256']==binding['recovery_sha256'] and restored['previous_contract']==binding['previous_contract']
 assert js(Q/'EPOCHS_B.json')[:100]==js(Path(binding['previous_root'])/'EPOCHS_B.json')
 assert sha(binding['completed_stage6_cp'])==binding['completed_stage6_sha256']
manifests={};stage_labels={}
for stage in range(6,10):
 manifest=js(O/('ra_coreset/dataset_manifests/task%d.json'%stage))['rows'];assert len(manifest)==POP[stage] and all(sha(r['path'])==r['sha256'] for r in manifest)
 labels={}
 for t in range(6,stage):
  p=O/('effects/memory/task%d.pth'%t);memory=torch.load(p,map_location='cpu');storage=js(O/('ra_coreset/storage/task%d.json'%t))['selected']
  md={Path(r['path']).name:r for r in manifest if r['task']==t};assert set(md)==set(memory)=={r['destination'] for r in storage}
  for r in storage:assert r['sha256']==md[r['destination']]['sha256']==memory[r['destination']]['sample_sha256']
  labels.update({str(O/'ra_coreset/buffer'/n):v for n,v in memory.items()})
 p=O/('effects/current/task%d/labels.pth'%stage);labels.update({str(O/'features'/str(stage)/n):v for n,v in torch.load(p,map_location='cpu').items()})
 aligned=[]
 for m in manifest:
  r=labels[m['path']];assert r['sample_sha256']==m['sha256'] and all(r[k]==m[k] for k in ('task','demo','u','bc_valid'))
  if r['bc_valid']:assert r['identity_unique'] and r['full_window_valid'] and r['next_action_index']==r['u']+1
  aligned.append(r)
 manifests[stage]=manifest;stage_labels[stage]=aligned
strict={};diags={};teacher_audits={};calls=0;windows=0;tic=time.time();task6={}
for arm in ARMS:
 strict[arm]={};hist=js(Q/('EPOCHS_'+arm+'.json'));assert len(hist)==400
 for stage in range(6,10):
  cp=model_path(arm,stage);p=model(cp);meta=js(Q/('STAGE_%s_%d.json'%(arm,stage)));assert sha(cp)==meta['sha256']
  raw=torch.load(cp,map_location='cpu');strict[arm][str(stage)]=dict(strict_keys=100,parameters=sum(v.numel() for v in p.parameters()),sha256=sha(cp),steps=meta['steps'])
  if stage==6:
   task6[arm]={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in raw['state_dict'].items()}
   baseline=torch.load(O/'ra_coreset/models/mlrifa_6_model_ep99.pth',map_location='cpu')['state_dict'];assert all(torch.equal(v,baseline[k]) for k,v in raw['state_dict'].items());continue
  tm=js(Q/('TEACHER_%s_%d.json'%(arm,stage)));tc=torch.load(teacher_path(arm,stage),map_location='cpu');assert tm['source_sha256']==sha(model_path(arm,stage-1))==tc['source_sha256'] and tm['cache_sha256']==sha(teacher_path(arm,stage))
  ids=[i for i,r in enumerate(manifests[stage]) if r['task']<stage and r['bc_valid']];assert ids==tc['ids'] and set(ids)==set(tc['cache']) and len(ids)==OLD[stage]
  targets=torch.stack([torch.as_tensor(stage_labels[stage][i]['next_action']) for i in ids]).float()
  scale=torch.stack([targets[:,a:b].square().mean().sqrt().clamp_min(.1) for a,b in [(0,3),(3,6),(6,7)]]).double();assert torch.allclose(tc['scales'].double(),scale,atol=1e-7,rtol=1e-7)
  ct=np.array([r['teacher_cost'] for r in tm['rows']]);assert abs(tm['tau']-max(1e-8,float(np.median(ct))))<1e-14
  assert all(abs(r['weight']-tm['tau']/(tm['tau']+r['teacher_cost']))<1e-7 for r in tm['rows'])
  teacher=model(model_path(arm,stage-1));out=[];maxcache=0.
  for start in range(0,len(ids),32):
   chunk=ids[start:start+32];x=torch.stack([torch.load(manifests[stage][i]['path'],map_location='cpu')['feature'].squeeze(0) for i in chunk]);y=torch.stack([torch.as_tensor(stage_labels[stage][i]['next_action']) for i in chunk]).double()
   tq=teacher(x);sq=p(x);calls+=2;windows+=2*len(chunk)
   cq=tuple(torch.stack([tc['cache'][i][j] for i in chunk]) for j in range(3))
   for u,v in zip(tq,cq):
    diff=float((u-v).abs().max());maxcache=max(maxcache,diff);assert torch.allclose(u,v,atol=2e-4,rtol=2e-4),'Cached teacher vs CPU checkpoint mismatch'
   tq=tuple(v.double() for v in tq);sq=tuple(v.double() for v in sq);tmus,tstd,tl=tq;smus,sstd,sl=sq;tk=tl.argmax(-1);sk=sl.argmax(-1);ii=torch.arange(len(chunk))
   def groups(mu,sd):return torch.stack([((mu[:,:,a:b]-y[:,None,a:b]).square()+sd[:,:,a:b].square()).mean(-1)/scale[g].square() for g,(a,b) in enumerate([(0,3),(3,6),(6,7)])],-1)
   tg=groups(tmus,tstd);sg=groups(smus,sstd);anchor=torch.stack([((smus[:,:,a:b]-tmus[ii,tk][:,None,a:b]).square()+(sstd[:,:,a:b]-tstd[ii,tk][:,None,a:b]).square()).mean(-1)/scale[g].square() for g,(a,b) in enumerate([(0,3),(3,6),(6,7)])],-1)
   for j,i in enumerate(chunk):
    w=tm['rows'][start+j];assert w['index']==i and w['sample_sha256']==manifests[stage][i]['sha256']
    out.append(dict(index=i,task=manifests[stage][i]['task'],sample_sha256=manifests[stage][i]['sha256'],teacher_mode=int(tk[j]),student_mode=int(sk[j]),teacher_component_group_costs=tg[j].tolist(),student_component_group_costs=sg[j].tolist(),student_component_anchor_group_distances=anchor[j].tolist(),student_logits=sl[j].tolist(),teacher_logits=tl[j].tolist(),quality_weight=w['weight'],frozen_teacher_cost=w['teacher_cost']))
  diags[arm+'_'+str(stage)]=out;teacher_audits[arm+'_'+str(stage)]=dict(count=len(ids),source_sha256=tm['source_sha256'],cache_sha256=tm['cache_sha256'],cache_cpu_max_abs=maxcache,tau=tm['tau'],scale=scale.tolist())
 rec=torch.load(OUTPUT[arm]/'recovery.pth',map_location='cpu');assert rec['stage']==9 and rec['next_epoch']==100 and rec['steps']==76100 and rec['state']['step']==76100 and rec['contract']==sha(Q/'CONTRACT.json') and rec['history']==hist
 assert (OUTPUT[arm]/'recovery.pth').stat().st_ino==model_path(arm,9).stat().st_ino
 assert {'model','optimizer','scheduler','grads','modes','requires','rng','step'}==set(rec['state']) and set(rec['state']['rng'])=={'python','numpy','cpu','cuda'}
 assert all(torch.equal(rec['state_dict'][k],v) for k,v in rec['state']['model'].items())
 norm=lambda names:{n[7:] if n.startswith('module.') else n for n in names}
 assert norm(rec['state']['grads'])==set(dict(p.named_parameters()))==norm(rec['state']['requires'])
 assert rec['teacher_sha256']==sha(teacher_path(arm,9)) and rec['teacher_metadata_sha256']==sha(Q/('TEACHER_%s_9.json'%arm))
 strict[arm]['recovery']=dict(sha256=sha(OUTPUT[arm]/'recovery.pth'),complete_state_fields=sorted(rec['state']),history_exact=True,final_hardlink=True)
assert all(v==task6['B'] for v in task6.values())
records=js(Q/'OUTCOMES.json');assert len(records)==800
cfg=js(O/'resolved_eval.json');sys.path.insert(0,str(S));from libero.libero.benchmark import get_benchmark
with contextlib.redirect_stdout(sys.stderr):b=get_benchmark(cfg['benchmark_name'])(cfg['data']['task_order_index'])
initial={}
for t in range(6,10):
 task=b.get_task(t);fp=Path(cfg['init_states_folder'])/task.problem_folder/task.init_states_file;arr=torch.load(fp,map_location='cpu');initial[t]=(sha(fp),[hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in arr[:20]])
seen=set()
for r in records:
 key=(r['arm'],r['stage'],r['task'],r['initial_index']);assert key not in seen;seen.add(key);assert r['attempt']==1 and 0<r['steps']<=600 and (r['success'] or r['steps']==600)
 assert initial[r['task']][0]==r['initial_file_sha256'] and initial[r['task']][1][r['initial_index']]==r['initial_sha256']
 noise=np.random.RandomState(20261024+41000+r['task']*100+r['initial_index']).standard_normal((600,7)).astype(np.float32)
 assert hashlib.sha256(noise.tobytes()).hexdigest()==r['noise_sha256'] and r['cp_sha256']==strict[r['arm']][str(r['stage'])]['sha256']
assert len(seen)==800;verify();light={}
for fp in Q.rglob('*'):
 if fp.is_file() and fp.suffix in ['.json','.jsonl','.exit'] and fp.name!='CONTRACT.json' and fp.stat().st_size<8*1024**2:light[str(fp.relative_to(Q))]=dict(text=fp.read_text(),sha256=sha(fp))
print(json.dumps(dict(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',strict=strict,contract=sha(Q/'CONTRACT.json'),contract_sha256=sha(Q/'CONTRACT.json'),diagnostics=diags,teacher_audits=teacher_audits,task6_equal_to_ALIGN_and_between_arms=True,diagnostic_calls=calls,diagnostic_windows=windows,diagnostic_seconds=time.time()-tic,light=light,episodes=800,environment_calls=0,optimizer_updates=0,no_residual=True,processes=[],observed=time.time())))
