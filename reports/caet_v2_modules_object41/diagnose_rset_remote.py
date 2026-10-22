"""Executed after the existing verified read-only gate data-loader prefix."""
import math,contextlib,traceback
print('Verified data loaded; diagnosing support',file=sys.stderr,flush=True)
def ident(r):return dict(sample=r['sample'],task=r['task'],demo=r['demo'],u=r['u'])
def packed(rows):
 return dict(rows=rows,d=torch.stack([r['descriptor'] for r in rows]),a=torch.stack([r['next_action'] for r in rows]),keys=[(r['task'],r['demo']) for r in rows])
def old_query(r,a,bank,scale):
 sim=(bank['d']@r['descriptor']).clamp(-1,1);valid=torch.tensor([k!=(r['task'],r['demo']) for k in bank['keys']]);dist=((bank['a']-a)/scale).square().mean(-1).sqrt()
 state=(sim>=.90)&valid;act=(dist<=.50)&valid;joint=state&act
 top=torch.zeros(len(sim),dtype=torch.bool);masked=sim.clone();masked[~valid]=-2;top[masked.topk(min(20,len(sim))).indices]=True
 def nd(mask):return len({bank['keys'][i] for i in torch.where(mask)[0].tolist()})
 counts=dict(state_full=nd(state),action_full=nd(act),joint_full=nd(joint),state_top20=nd(state&top),action_top20=nd(act&top),joint_top20=nd(joint&top))
 # A best joint candidate per demo retains every demo having any qualifying point.
 counts['per_demo_joint']=counts['joint_full']
 return dict(demos=counts,passed={k:v>=3 for k,v in counts.items()},state_any=bool(state.any()),action_any=bool(act.any()),max_cos=float(sim[valid].max()),minimum_action_rms=float(dist[valid].min()))
def components(bank,desc,a,scale,exclude=None):
 sim=(desc@bank['d'].T).clamp(-1,1);state=(2-2*sim).clamp_min(0).sqrt()
 arm=((a[:,None,:6]-bank['a'][None,:,:6])/scale[:6]).square().mean(-1).sqrt()
 grip=(a[:,None,6]-bank['a'][None,:,6]).abs()/scale[6]
 match=torch.sign(a[:,None,6])==torch.sign(bank['a'][None,:,6]);sq=state.square()+arm.square()+grip.square()
 if exclude is not None:
  for j,key in enumerate(exclude):match[j]&=torch.tensor([k!=key for k in bank['keys']])
 sq[~match]=float('inf');best=[]
 for key in sorted(set(bank['keys'])):
  idx=torch.tensor([i for i,k in enumerate(bank['keys']) if k==key]);v,ii=sq[:,idx].min(-1);actual=idx[ii];jj=torch.arange(len(a))
  best.append(torch.stack([v,state[jj,actual],arm[jj,actual],grip[jj,actual]],-1))
 z=torch.stack(best,1);order=z[:,:,0].argsort(-1)[:,:min(3,len(best))];sel=torch.gather(z,1,order[:,:,None].expand(-1,-1,4));finite=torch.isfinite(sel[:,:,0]);n=finite.sum(-1)
 result=[]
 for j in range(len(a)):
  if n[j]<2:result.append(None);continue
  val=sel[j,finite[j]];result.append(dict(d=float(val[:,0].mean().sqrt()),state=float(val[:,1].square().mean().sqrt()),arm=float(val[:,2].square().mean().sqrt()),gripper=float(val[:,3].square().mean().sqrt()),demos=int(n[j])))
 return result
def quant(rows,key,q):
 valid=[(x[key],demo) for x,demo in rows if x is not None];counts=collections.Counter(d for _,d in valid)
 assert valid
 data=sorted((v,1./counts[d]/len(counts)) for v,d in valid);c=0
 for v,w in data:
  c+=w
  if c>=q-1e-12:return v
 return data[-1][0]
cache={}
def estimator(task,exclude):
 key=(task,exclude)
 if key in cache:return cache[key]
 rows=[r for r in allrows[task] if not r['holdout'] and r['demo']!=exclude];bank=packed(rows);scale=bank['a'].std(0,unbiased=False).clamp_min(torch.tensor([.001]*6+[.05]))
 cal=[]
 for demo in sorted({r['demo'] for r in rows}):
  train=[r for r in rows if r['demo']!=demo];b=packed(train);s=b['a'].std(0,unbiased=False).clamp_min(torch.tensor([.001]*6+[.05]))
  query=[r for r in rows if r['demo']==demo]
  for st in range(0,len(query),64):
   part=query[st:st+64];vals=components(b,torch.stack([r['descriptor'] for r in part]),torch.stack([r['next_action'] for r in part]),s)
   cal.extend((v,demo) for v in vals)
 bounds=dict(tau=max(quant(cal,'d',.5),1e-6),d=quant(cal,'d',.95),state=quant(cal,'state',.99),arm=quant(cal,'arm',.99),gripper=quant(cal,'gripper',.99))
 meta=dict(excluded_demo=exclude,training_demos=sorted({r['demo'] for r in rows}),calibration_count=len(cal),calibration_estimable=sum(v is not None for v,d in cal),bounds=bounds,scale=scale.tolist(),calibration=[dict(demo=d,metrics=v) for v,d in cal])
 cache[key]=(bank,scale,bounds,meta);return cache[key]
def local_query(rows,actions,task):
 output=[None]*len(rows)
 for demo in sorted({r['demo'] for r in rows}):
  ix=[i for i,r in enumerate(rows) if r['demo']==demo];exclude=None if rows[ix[0]]['holdout'] else demo
  b,s,bounds,meta=estimator(task,exclude)
  for st in range(0,len(ix),64):
   ids=ix[st:st+64];v=components(b,torch.stack([rows[i]['descriptor'] for i in ids]),actions[ids],s)
   for i,m in zip(ids,v):
    why=['INSUFFICIENT_DISTINCT_DEMOS'] if m is None else [k for k in ('d','state','arm','gripper') if m[k]>bounds[k]]
    output[i]=dict(metrics=m,reasons=why,accepted=not why,weight=math.exp(-m['d']**2/bounds['tau']**2) if not why else 0.,calibration_excluded_demo=exclude)
 return output
def summarize(rows):
 n=len(rows);return dict(n=n,local_accepted=sum(x['local']['accepted'] for x in rows),weight_sum=sum(x['local']['weight'] for x in rows),cross_local=sum(x['cross'] and x['local']['accepted'] for x in rows),cross_weight_sum=sum(x['local']['weight'] for x in rows if x['cross']),cross=sum(x['cross'] for x in rows),old_pass={bank:{k:sum(x['old'][bank]['passed'][k] for x in rows) for k in rows[0]['old'][bank]['passed']} for bank in rows[0]['old']} if rows else {})
result=dict(status='ZERO_TRAINING_DIAGNOSTIC',protocol_sha256=REQUEST['diagnostic_protocol_sha256'],seed=41,stages={},models=[],optimizer_calls=0,environment_calls=0,server_writes=0,model_forward_calls=0,predictor_gate='NOT_TESTED',input_evidence_sha256=REQUEST['input_evidence_sha256'])
for stage in (8,9):
 cur=[r for r in allrows[stage] if not r['holdout']];old=[r for t in range(6,stage) for r in oldrows[t] if not r['holdout']]
 # Bind cached graph identities rather than rerunning frozen scientific graph.
 g=REQUEST['stages'][str(stage)];cross={g['current_nodes'][i]['sample'] for i,j in g['edges']};assert len(cross)==g['supported_current']
 mixed=packed(cur+old);same=packed(cur);ms=mixed['a'].std(0,unbiased=False).clamp_min(.05);ss=same['a'].std(0,unbiased=False).clamp_min(.05)
 query=allrows[stage];new=local_query(query,torch.stack([r['next_action'] for r in query]),stage);e=[]
 for i,r in enumerate(query):
  e.append(dict(identity=ident(r),holdout=r['holdout'],cross=r['sample'] in cross,local=new[i],old={name:old_query(r,r['next_action'],b,s) for name,b,s in [('all_task',mixed,ms),('current_task_mixed_scale',same,ms),('current_task_own_scale',same,ss)]}))
 original={x['sample']:x for x in g['rset_zero_training']['queries']}
 assert all(x['old']['all_task']['passed']['joint_top20']==original[x['identity']['sample']]['expert_action_supported'] for x in e if not x['holdout'])
 result['stages'][str(stage)]=dict(expert=e,summary={name:summarize([x for x in e if cond(x)]) for name,cond in [('all_train',lambda x:not x['holdout']),('effect_supported_train',lambda x:x['cross']),('effect_holdout',lambda x:x['holdout'])]},exact_original_recomputed=True)
 print('Stage %s expert diagnostic complete'%stage,file=sys.stderr,flush=True)
# Freeze policy audit: CPU native architecture; no constructors that create optimizers or output directories.
try:
 S=Path('/data/mlr_ifa/experiment_workspaces/CAETv2_TAC_RSET_EFTR_OBJECT41_20261022/RSET/source');sys.path.insert(0,str(S))
 with contextlib.redirect_stdout(sys.stderr):
  from libero.lifelong.models.brief import BriefPolicy
  from libero.lifelong.models.modules.transformer_modules import OfficialGPT2Decoder
  from libero.lifelong.models.policy_head import GMMHead
 modelroot=Path('/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed41_20261021')
 for stage in (8,9):
  cur=[r for r in allrows[stage] if not r['holdout']];old=[r for t in range(6,stage) for r in oldrows[t] if not r['holdout']];mixed=packed(cur+old);same=packed(cur);ms=mixed['a'].std(0,unbiased=False).clamp_min(.05);ss=same['a'].std(0,unbiased=False).clamp_min(.05)
  g=REQUEST['stages'][str(stage)];cross={g['current_nodes'][i]['sample'] for i,j in g['edges']};queries=[r for r in cur if r['sample'] in cross]
  for checkpoint_task in (stage-1,stage):
   cp=modelroot/'ra_coreset/models'/('mlrifa_%d_model_ep99.pth'%checkpoint_task);digest=sha(cp);raw=torch.load(cp,map_location='cpu');assert raw['task']==checkpoint_task and raw['epoch_index']==99
   torch.manual_seed(20261022)
   policy=BriefPolicy.__new__(BriefPolicy);torch.nn.Module.__init__(policy)
   policy.temporal_transformer=OfficialGPT2Decoder(embed_size=512,num_layers=6,num_heads=8,dropout=.15,max_position_embeddings=32,train=True)
   policy.policy_head=GMMHead(input_size=512,output_size=7,num_modes=raw['cfg']['policy']['num_gmm_modes'],min_std=1e-4)
   sd={k[7:] if k.startswith('module.') else k:v for k,v in raw['state_dict'].items()};assert len(sd)==100
   policy.load_state_dict(sd,strict=True);policy.eval();policy.requires_grad_(False);means=[];samples=[]
   gen=torch.Generator().manual_seed(20261022)
   with torch.no_grad():
    for st in range(0,len(queries),16):
     part=queries[st:st+16];x=torch.stack([torch.load(P/'features'/str(stage)/r['sample'],map_location='cpu')['feature'].squeeze(0) for r in part]);z=policy.temporal_encode(x)[:,-1];m,s,l=policy.policy_head.forward_fn(z);ids=l.argmax(-1);jj=torch.arange(len(ids));m=m[jj,ids];s=s[jj,ids];means.append(m);samples.append(m+s*torch.randn(m.shape,generator=gen));result['model_forward_calls']+=1
   audit=dict(stage=stage,checkpoint_task=checkpoint_task,checkpoint=str(cp),sha256=digest,strict_keys=len(sd),role='before_stage' if checkpoint_task==stage-1 else 'after_stage',queries=len(queries),actions={})
   for name,act in [('argmax_component_mean',torch.cat(means)),('argmax_component_Normal_seed20261022',torch.cat(samples))]:
    assert torch.isfinite(act).all();vals=local_query(queries,act,stage);rows=[]
    for i,r in enumerate(queries):rows.append(dict(identity=ident(r),cross=True,local=vals[i],old={n:old_query(r,act[i],b,s) for n,b,s in [('all_task',mixed,ms),('current_task_mixed_scale',same,ms),('current_task_own_scale',same,ss)]}))
    audit['actions'][name]=dict(rows=rows,summary=summarize(rows))
   assert sha(cp)==digest;audit['unchanged']=True;result['models'].append(audit);del policy,raw,sd
   print('Frozen CPU policy stage %s CP%s complete'%(stage,checkpoint_task),file=sys.stderr,flush=True)
 result['policy_status']='PASS'
except Exception:
 result['policy_status']='INCOMPLETE';result['policy_error']=traceback.format_exc()
result['calibrations']={str(t)+'/'+str(d):v[3] for (t,d),v in cache.items()}
result['input_bindings']={t:dict(binding_sha256=x['binding_sha256'],storage_sha256=x['storage_sha256']) for t,x in out['tasks'].items()}
result['cuda_initialized']=torch.cuda.is_initialized();assert not result['cuda_initialized'];result['finished_unix']=time.time();print(json.dumps(result))
