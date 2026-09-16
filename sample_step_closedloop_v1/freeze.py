import gzip,time,sys,platform,subprocess
from .common import *
def run():
 assert not OUT.exists();OUT.mkdir(parents=True)
 assert read(PRIOR/'INDEPENDENT_AUDIT.json')['status']=='PASS'
 from cb_ifa_experiment.gpu_runtime import torch,digest
 from gradient_relation_experiment.prepare import batch_indices
 from libero.libero.benchmark import get_benchmark
 assert torch.cuda.get_device_name(0)==DEVICE
 old=read(PRIOR/'PROTOCOL_FROZEN.json')
 with gzip.open(ROOT/'reports/gradient_relation_4090_v2/DESIGN.json.gz','rt') as f:design=json.load(f)
 pts={x['point']:x for x in design['points']}
 names=[next(n for n in old['points'] if n.startswith(f's{s}_t8_')) for s in [45,46,47]]
 members={r['point']:r for r in read(ROOT/'reports/sensitive_component_compatibility_v1/PROBE_MEMBERSHIP.json')['points']}
 lineage=read(ROOT/'reports/paper_active_set_v1/PROBE_WINDOW_MEMBERSHIP.json')['windows']
 origins={x['point']:x for s in [45,46,47] for x in rows(ROOT/f'reports/functional_geometry_diagnostic_v1/carrier/seed{s}/origins.jsonl')}
 batches={};anchors={};files=set();configs=[]
 for name in names:
  pt=pts[name];seed=pt['seed'];spec=design['source_plan_sha256'][f'{seed}/8'];plan=read(ROOT/spec['path']);assert sha(ROOT/spec['path'])==spec['sha256']
  sequence=[batch_indices(plan,pt['stage_step']+i) for i in range(300)]
  assert sequence[:20]==pt['future20_record_indices'] and all(0<len(x)<=10 for x in sequence)
  batches[name]=dict(point=pt,plan=spec,batches=sequence,origin=origins[name],probe=members[name]);a=origins[name]['anchor'];assert sha(a['path'])==a['sha256'];anchors[name]=dict(a)
  for f,h in zip(plan['immutable_feature_origins'],plan['record_sha256']):assert sha(ROOT/f)==h;files.add(ROOT/f)
  for r in lineage:
   if r['point']==name and r['eval_task']<8:assert r['source']=='actual_immutable_buffer'
  cfg=eval_cfg(seed,OUT);assert cfg.policy.use_language and not cfg.policy.use_state and cfg.eval.max_steps==600
  configs.append(digest([dict(cfg.policy),dict(cfg.data),dict(cfg.eval)]));write(OUT/f'EVAL_CONFIG_seed{seed}.json',cfg)
  bench=get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
  for t in [6,7,8]:
   task=bench.get_task(t);f=Path(cfg.init_states_folder)/task.problem_folder/task.init_states_file;assert len(torch.load(f,map_location='cpu'))>=50;files.add(f)
 assert len(set(configs))==1
 write(OUT/'TRAINING_MEMBERSHIP.json',batches);write(OUT/'ORIGINAL_ANCHORS.json',anchors)
 write(OUT/'PROBE_LINEAGE.json',[r for r in lineage if r['point'] in names])
 cells=[];orders=[ARMS,ARMS[1:]+ARMS[:1],ARMS[2:]+ARMS[:2]]
 for i,name in enumerate(names):
  seed=pts[name]['seed']
  for task in [6,7,8]:
   for arm in orders[i]:cells.append(dict(point=name,task=task,arm=arm,environment_seed=seed,round_policy_seeds=[9310000+i*1000+task*10+r for r in range(5)]))
 p=dict(name='sample_step_closedloop_v1',time=time.time(),points=names,seeds=[45,46,47],arms=ARMS,training_steps=300,cells=cells,train_orders=orders,
  origin_selection='First task8 origin in prior frozen order for each seed, including seed47. No outcome-based selection.',
  training='300 consecutive original mixed batches from the true original3090 anchor, faithfully restored on validated4080S. Original BC, weights, frontend, clip, AdamW, scheduler and epoch RNG semantics. No reset of moments.',
  references='Same fixed32 existing P buffer windows per old task6/7 for both QP arms, temporal windows not independent demos. Gradients recomputed at each current branch state on every update. TaskQP uses one mean/task; SampleQP uses64 per-window rows jointly. No new samples, no Fisher/rank/slack/kappa. Fixed reference reuse is a limitation.',
  update='For actual native AdamW displacement d, solve min .5||v-d||^2 with h_i^T v<=0. Commit native moments/scheduler and parameters theta+v. Each branch evolves its own true moments thereafter. No sequential projection.',
  evaluation='Only fixed step300 endpoint, tasks6/7/8,50 trials/cell,10lanes,5rounds,originalGMM sampling,600steps,5warmup steps,history8,native successOR. 27cells/1350trials; no best checkpoint/trial extension.',
  pairing='Same initial indices0..49 and task-specific round policy seeds across arms; environment seed=trainingseed+round. Initial observation hashes must match. Model action draws remain stochastic with common seeds, not forced equal actions.',
  primary='Equal-task mean success across6/7/8 per seed, then equal3seed mean. Predeclared contrasts SAMPLE_STEP_QP minus MLR and minus TASK_STEP_QP.',
  secondary=['old6/7balanced success','newtask8 success','each task and seed','worst old task','paired wins/losses/ties','training correction norms/rates and compute'],
  decisions=dict(candidate='Both primary contrasts >=2pp, each positive in>=2seeds; versusMLR oldmean>0, newmean>=-2pp; no seed primary<=-5pp, no seed new<=-5pp; no task degrades>=10pp in>=2seeds.',
   not_supported='Sample-minusMLR pooled primary<=0pp, or pooled newtask<=-5pp.',otherwise='INCONCLUSIVE',status='Development screening thresholds; not theoretical or significance guarantees.'),
  uncertainty='Report all3development seeds and individual paired cells.50initial states repeated acrossseeds/arms are not1350independent training replicates. No seed-population significance claims.10000 CPU bootstrap draws using private seed20260916, resampling50initial-state indices within task and linking indices acrossall seeds/arms, produces only conditional descriptive95percentile intervals.',
  engineering='CPUtests; all3origin one-step state hashes exactly match prior MLR/taskSTEP/sampleSTEP. Each arm serialize/restore and reproduce step2 bitwise, including unchanged native moments after displacement correction. Duplicate identical-model50trial evaluation with tracing on/off, exact actions/dones/RNG/observations. No effects used to tune.100engineering trials excluded.',
  monitoring='No A/B bank used for selection or reported as independent300-step monitoring. Primary is actual closed-loop success. Training probe effects are not validation.',
  resume='Atomic full state every25steps; interrupted branch not automatically retried. Preserve failures and completed results; only documented technical resume with identical sequence. No score-based retries or early stop.',
  limitations='Only300updates at task8,3development seeds, one origin/seed, fixed referencepool,method selected on earlier single-step BC. Not full lifelong or independent confirmation.',
  resources=dict(GPU=DEVICE,one_worker=True,wall_hours=72,train_job_hours=12,eval_cell_hours=2,host_rss_GiB=28,host_available_min_GiB=2,cuda_allocated_GiB=20,cuda_reserved_GiB=22,data_free_min_GiB=1,root_free_min_GiB=8,new_reports_max_GiB=.75,new_states_max_GiB=8))
 write(OUT/'PROTOCOL_FROZEN.json',p)
 for rel,h in read(PRIOR/'FROZEN_INPUTS.json').items():assert sha(ROOT/rel)==h;files.add(ROOT/rel)
 for pkg in ['sample_step_closedloop_v1','paper_qp_rollout_v1','optimizer_signal_diagnostic']:
  files.update((ROOT/pkg).glob('*.py'))
 for sub,glob in [('libero/lifelong','*.py'),('libero/configs','*.yaml'),('libero/libero/envs','*.py'),('libero/libero/bddl_files/libero_object','*')]:files.update((ROOT/sub).rglob(glob))
 files.add(ROOT/'checkpoints/object_multitask_model_ep99.pth');files.update(OUT.glob('*.json'))
 for n in names:
  files.update((PRIOR/'measurements'/n).glob('*STATE_HASHES.json'))
  seed=pts[n]['seed'];files.add(ROOT/f'reports/functional_geometry_diagnostic_v1/carrier/seed{seed}/native_witnesses.jsonl')
 write(OUT/'FROZEN_INPUTS.json',{f.relative_to(ROOT).as_posix():sha(f) for f in sorted(files) if f.is_file()})
 write(OUT/'ENVIRONMENT.json',dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],text=True),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),platform=platform.platform()))
 (OUT/'PROTOCOL.md').write_text('# 三seed逐样本位移QP配对闭环筛查\n\n三个原始task8起点；三版本分别连续300步，固定step300评估task6/7/8，每任务50配对trial，总1350trial。\n\n主比较：逐样本位移QP对MLR、对任务平均位移QP。共享原始状态与批次，固定旧任务32窗口/任务，各分支每步重新求梯度并联合求解，提交原生AdamW moments。\n\n主指标为任务等权成功率，单列旧任务和新任务；保留seed47，禁止按结果换seed、选checkpoint、追加trial或调规则。本轮是开发筛查，不是完整终身训练或独立确认。\n\n所有数值、名单、采样、决策阈值见PROTOCOL_FROZEN.json与TRAINING_MEMBERSHIP.json。A/B不作为300步隔离证据。\n',encoding='utf-8')
 print('PROTOCOL_FROZEN',names,len(cells),flush=True)
if __name__=='__main__':run()
