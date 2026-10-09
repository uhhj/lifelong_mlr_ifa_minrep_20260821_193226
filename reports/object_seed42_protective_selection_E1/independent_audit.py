"""Read-only experiment audit. Never trains or constructs an environment."""
import argparse,collections,csv,datetime,hashlib,json,math,random,re,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--server',action='store_true');a=p.parse_args()
O=Path(a.root);D=Path(a.output);D.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def metrics(m):
    f=sum(m[i][i] for i in range(4))/4
    n=sum(sum(m[i][i]-m[j][i] for j in range(i+1,4))/(3-i) for i in range(3))/3
    u=[sum(m[j][i] for j in range(i,4))/(4-i) for i in range(4)]
    return dict(auc=sum(u)/4,fwt=f,nbt=n,task_auc=u)
receipts={};matrix={};chosen={};allcells=[]
for view,target in [('primary',200),('secondary',680)]:
    path=O/('trials_'+view+'.jsonl');rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows)==target and sha(path)==read(O/(view+'_COMPLETE.json'))['trials_sha256']
    g=collections.defaultdict(list)
    for r in rows:
        assert r['seed']==42 and type(r['success']) is bool
        assert Path(r['checkpoint']).name=='mlrifa_%d_model_ep%d.pth'%(r['stage'],r['epoch'])
        assert 6<=r['task']<=r['stage']<=9 and 1<=r['vector_group_steps']<=600
        g[r['stage'],r['epoch'],r['task']].append(r)
    for (s,e,t),rs in g.items():
        assert len(rs)==20 and sorted(x['initial_state_index'] for x in rs)==list(range(20))
        assert sorted(x['trial'] for x in rs)==list(range(20))
        allcells.append(dict(view=view,stage=s,epoch=e,task=t,n=20,successes=sum(x['success'] for x in rs)))
    rate={k:sum(x['success'] for x in rs)*5 for k,rs in g.items()}
    if view=='primary':
        expected={(s,99,t) for s in range(6,10) for t in range(6,s+1)};best={s:99 for s in range(6,10)}
    else:
        epochs=[70,75,80,85,90,95,99]
        best={s:max(epochs,key=lambda e:(rate[s,e,s],e)) for s in range(6,10)}
        expected={(s,e,s) for s in range(6,10) for e in epochs}|{(s,best[s],t) for s in range(6,10) for t in range(6,s)}
    assert set(g)==expected
    m=[[rate[s,best[s],t] if t<=s else None for t in range(6,10)] for s in range(6,10)]
    emitted=read(O/('metrics_'+view+'.json'));assert m==emitted['r']
    for k in ('auc','fwt','nbt'):assert round(metrics(m)[k],3)==emitted[k]
    assert {str(k):v for k,v in best.items()}==emitted['selected_epochs']
    matrix[view]=m;chosen[view]=best;receipts[view]=dict(records=len(rows),cells=len(g),sha256=sha(path))
training=read(O/'TRAINING_COMPLETE.json');curve=list(csv.DictReader((O/'run_001/paper_repro_train_curve.csv').open()))
lr=list(csv.DictReader((O/'run_001/lr_protocol.csv').open()));assert len(curve)==len(lr)==400
counts={6:157,7:227,8:309,9:362};steps=0;diagnostics={};parity=read(O/'FEATURE_PARITY.json');previous={}
for t,b in counts.items():
    rs=[r for r in curve if int(r['stage'])==t];ls=[r for r in lr if int(r['stage'])==t]
    assert sorted(int(r['epoch']) for r in rs)==list(range(100))
    assert all(int(r['batches'])==b and int(r['batch_size'])==10 for r in rs)
    for r in ls:
        e=int(r['epoch']);assert int(r['optimizer_step_start'])==e*b and int(r['optimizer_step_end'])==(e+1)*b
        assert int(r['warmup_steps'])==500 and r['scheduler_mode']=='batch_step' and int(r['total_steps'])==100*b
    steps+=100*b
    cat=read(O/('catalog_task%d.json'%t))
    assert {k[7:]:v['sha256'] for k,v in cat.items() if k.startswith('buffer/')}==previous
    current={k for k in cat if k.startswith('current/')};assert len(current)==len(parity['features'][str(t)])
    selected=random.Random(42).sample(sorted(parity['features'][str(t)]),math.ceil(len(current)*.5))
    previous.update({'t%d__%s'%(t,n):parity['features'][str(t)][n] for n in selected})
    assert previous==read(O/('buffer_after_task%d.json'%t))
    events=[]
    for f in sorted((O/'selection_logs'/('task%d'%t)).glob('epoch*.jsonl')):
        events.extend(json.loads(x) for x in f.read_text().splitlines())
    assert len(events)==100*b and [e['stage_optimizer_step'] for e in events]==list(range(1,100*b+1))
    refresh=[];active={};used=collections.Counter();applied=0;veto=0
    for e in events:
        n=e['stage_optimizer_step'];assert e['stage']==t and e['epoch']==(n-1)//b
        r=e['refresh']
        if r:
            refresh.append(r['step']);assert r['step']==n-1 and r['reason']=='OK'
            assert len(r['current_reference'])==16 and set(r['current_reference'])<=current
            active=r['valid_ids']
            for task,ids in r['candidates'].items():
                h=r['references'][task];assert len(h)==16 and len(ids)==64 and not set(h)&set(ids)
                assert len(set(ids))==64 and len(set(h))==16
                assert all(x in cat and cat[x]['task']==int(task)<t and x.startswith('buffer/') for x in h+ids)
                assert not r['invalid_candidates'][task] and active[task]==ids
                score=r['scores'][task];probs=r['probabilities'][task];weights=[math.exp(x-max(score)) for x in score]
                assert all(math.isfinite(x) for x in score) and abs(sum(probs)-1)<1e-10
                assert all(abs(p-w/sum(weights))<1e-10 for p,w in zip(probs,weights))
        for x in e['replacements']:
            assert 0<=x['slot']<10 and x['task']<t and x['candidate'] in active[str(x['task'])]
            assert cat[x['candidate']]['task']==x['task']
            if x['applied']:used[x['candidate']]+=1;applied+=1
            else:veto+=1;assert x['slot']==0 and x['reason']=='IFA_LANGUAGE_REFERENCE'
    assert refresh==([] if t==6 else list(range(0,100*b,500)))
    totals=training['selection_totals'][str(t)]
    assert totals['assigned_slots']==applied and totals['first_reference_veto']==veto and totals['refreshes']==len(refresh)
    diagnostics[t]=dict(refreshes=len(refresh),replacement_assignments=applied,old_slots=totals['old_slots'],
        assignment_fraction=applied/totals['old_slots'] if totals['old_slots'] else None,
        unique_priority_ids=len(used),repeat_assignments=sum(v-1 for v in used.values()),max_assignments_one_id=max(used.values(),default=0),
        top_reused=used.most_common(10),buffer_after=len(previous),veto=veto)
assert steps==105500
baseline=read(O/'FINAL_ACCEPTANCE.json')['baseline'];base_metrics=metrics(baseline['r']);e1=metrics(matrix['primary'])
delta={k:e1[k]-base_metrics[k] for k in ('auc','fwt','nbt')};assert delta['auc']<0
receipt=dict(status='PASS',audited_unix=time.time(),primary=receipts['primary'],secondary=receipts['secondary'],
 matrix=matrix,chosen_epochs=chosen,baseline_exact=base_metrics,e1_exact=e1,difference_exact=delta,
 secondary_exact=metrics(matrix['secondary']),training_epochs=400,optimizer_steps=steps,selection_diagnostics=diagnostics,
 all_stage_buffer_membership_reconstructed=True,condition_for_additional_seeds=False,
 limitations=['Task-slot/current-payload and IFA-reference invariants are enforced in frozen runtime and preflight, not reconstructible from per-step replacement events alone.',
 'No initial-state content hashes or individual action traces in inherited evaluator; records identify initial state index and checkpoint path. Their closure is supported by unchanged evaluator and full input/checkpoint hashes, not new trace claims.',
 'No original E0 stage buffer snapshots or wall-time logs; RTX4090 versus RTX3090 hardware difference remains. Single seed is not evidence of stable efficacy.'])
if a.server:
    S=Path('/data/mlr_ifa/experiment_workspaces/E1_protective_object_20261008/source')
    c=read(O/'CONTRACT.json');assert sha(O/'CONTRACT.json')=='b82fc3d9a58d82302924c0b48eda56e95361227776d59f1f41505dda50b4ee31'
    changed={k for k,v in c['source_sha256'].items() if sha(S/k)!=v}
    assert changed<={'.hydra/config.yaml','.hydra/hydra.yaml','.hydra/overrides.yaml'}
    for t,files in parity['features'].items():assert all(sha(O/'features'/t/n)==h for n,h in files.items())
    assert all(sha(O/'buffer'/n)==h for n,h in previous.items())
    assert sha(Path('/data/mlr_ifa/checkpoints/object_multitask_model_ep99.pth'))==c['pretrain_sha256']
    assert all(sha(O/'run_001'/n)==h for n,h in training['checkpoints'].items())
    assert len(training['checkpoints'])==28
    import torch
    shapes={}
    for task in range(6,10):
        cp=torch.load(O/'run_001'/('mlrifa_%d_model_ep99.pth'%task),map_location='cpu')
        state=cp['state_dict'];assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v) and v.is_floating_point())
        layout={k:list(v.shape) for k,v in state.items() if torch.is_tensor(v)}
        if shapes:assert layout==shapes
        shapes=layout
        del cp,state
    r=O/'engineering_recovery/20261008_import_binding'
    assert (r/'resume.exit').read_text().strip()=='0'
    processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    active=[s for s in processes.splitlines() if re.search(r'python\s+-m\s+protective_e1\.(evaluate|train)|bash\s+.*import_binding/resume_eval.sh',s)]
    assert not active,active
    gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
    receipt['server_checks']=dict(feature_hashes=5921,buffer_hashes=len(previous),checkpoint_hashes=28,
        final_checkpoints_cpu_deserialized=4,finite_parameters=True,same_parameter_shapes=True,
        source_inputs_unchanged=True,generated_hydra_differences=sorted(changed),recovery_exit=0,active_processes=active,
        gpu_available_at_audit=gpu.returncode==0,gpu_audit_message=(gpu.stdout+gpu.stderr).strip(),
        note='CPU deserialization only for file integrity, no policy/environment execution or CPU evaluation fallback.')
(D/'INDEPENDENT_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
with (D/'all_evaluation_cells.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(allcells[0]));w.writeheader();w.writerows(allcells)
print(json.dumps({k:receipt[k] for k in ('status','difference_exact','training_epochs','optimizer_steps','condition_for_additional_seeds')},indent=2))
