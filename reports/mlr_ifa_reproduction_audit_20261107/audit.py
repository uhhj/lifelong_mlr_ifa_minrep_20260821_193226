"""Read existing JSON/text only. No torch, models, training, environments or network."""
from pathlib import Path
from fractions import Fraction as F
import datetime, hashlib, json, math, statistics, tarfile

HERE = Path(__file__).resolve().parent
W = HERE.parents[1]
sources = {}

def raw(p):
    p = Path(p)
    if not p.is_absolute(): p = W / p
    b = p.read_bytes()
    sources[str(p.relative_to(W)).replace('\\','/')] = hashlib.sha256(b).hexdigest()
    return b

def js(p): return json.loads(raw(p).decode('utf-8-sig'))
def lines(p): return [json.loads(s) for s in raw(p).decode('utf8').splitlines() if s.strip()]
def metrics(r):
    return dict(FWT=sum(r[j,j] for j in range(6,10))/4,
        NBT=sum(sum(r[j,j]-r[i,j] for i in range(j+1,10))/(9-j) for j in range(6,9))/3,
        AUC=sum(sum(r[i,j] for i in range(j,10))/(10-j) for j in range(6,10))/4,
        OldFinal=sum(r[9,j] for j in range(6,9))/3,
        FinalAll=sum(r[9,j] for j in range(6,10))/4)
def display(m): return {k:dict(fraction=str(v),percent=float(v*100)) for k,v in m.items()}
def aggregate(ms):
    return {k:dict(mean_percent=float(sum(m[k] for m in ms)/len(ms)*100),
                   sample_sd_percent=statistics.stdev(float(m[k]*100) for m in ms)) for k in ms[0]}

ROOTS = {
 'OBJECT_41':'20261016_seeds40_41_three_arm/completion_evidence/seed41/rc',
 'OBJECT_42':'20261013_coreset_extension/final_evidence/remote/seed42',
 'OBJECT_43':'20261016_seed43_three_arm_repeat_v1/completion_evidence/RC',
 'GOAL_41':'20261019_ra_coreset_goal_seed41/progress_20261019_222601/completion_baseline',
 'GOAL_42':'20261017_coreset_goal_three_seed/final_evidence/seed42',
 'GOAL_43':'20261017_coreset_goal_three_seed/final_evidence/seed43',
 'OBJECT_43_FIRST':'20261013_coreset_extension/final_evidence/remote/seed43',
}
source_candidates = [W/'execution'/x/'coreset_exp' for x in [
 '20261013_coreset_object/source','20261013_coreset_extension/seed43/source',
 '20261017_coreset_goal_three_seed/seed43/source',
 '20261019_ra_coreset_goal_seed41/source']]
results={}; exact={}; baseline={s:js(f'execution/20261025_ccbp_v4_full/{s}_THREE_SEED_BASELINES.json') for s in ('OBJECT','GOAL')}
for case, rel in ROOTS.items():
    suite,seed=case.split('_')[:2];seed=int(seed)
    root=W/'execution'/rel
    modal=js(root/'MODAL_CONTRACT.json');accept=js(root/'INDEPENDENT_ACCEPTANCE.json')
    assert accept['status']=='PASS' and modal['flags']==dict(vision=True,language=True,state=True)
    cf=root/'resolved_random.json'
    if not cf.exists(): cf=W/'execution/20261016_seed43_three_arm_repeat_v1/launch_evidence/rc/resolved_random.json'
    cfg=js(cf);evcfg=js(root/'random/evaluation/resolved_actual.json')
    assert cfg['seed']==seed and cfg['gamma']==(.3 if suite=='OBJECT' else .7)
    assert cfg['train']['n_epochs']==100 and cfg['train']['batch_size']==10
    assert cfg['train']['optimizer']['kwargs']['lr']==1e-4
    assert cfg['linear_schedule_unit']=='optimizer_step'
    assert cfg['policy']['use_state'] and evcfg['policy']['use_state']
    assert evcfg['eval']['n_eval']==20 and evcfg['eval']['max_steps']==600
    assert evcfg['policy']['policy_head']['network_kwargs']['low_eval_noise'] is False
    # Bind actual executable training/model/evaluation sources to recorded hashes.
    bindings={}
    for name in ('train.py','model.py','evaluate.py','eval_metric.py'):
        want=modal['source_sha256']['coreset_exp/'+name]
        matches=[p/name for p in source_candidates if (p/name).exists() and hashlib.sha256((p/name).read_bytes()).hexdigest()==want]
        assert matches,(case,name,'No local source matching execution contract')
        raw(matches[0]);bindings[name]=str(matches[0].relative_to(W)).replace('\\','/')
    epochs=js(root/'random/epochs.json');complete=js(root/'random/TRAINING_COMPLETE.json')
    expected_batches=[157,227,309,362] if suite=='OBJECT' else [100,139,185,337]
    assert len(epochs)==400 and complete['epochs']==400 and complete['steps']==sum(expected_batches)*100
    step=0;stages={}
    def lr(n,total):return 1e-4*(n/500 if n<500 else max(0,(total-n)/(total-500)))
    for t,batches in zip(range(6,10),expected_batches):
        rr=[x for x in epochs if x['task']==t];assert [x['epoch'] for x in rr]==list(range(100))
        for e,x in enumerate(rr):
            step+=batches;assert x['steps']==step and x['batches']==batches and math.isfinite(x['loss'])
            assert math.isclose(x['lr_first'],lr(e*batches,100*batches),abs_tol=1e-14)
            assert math.isclose(x['lr_after'],lr((e+1)*batches,100*batches),abs_tol=1e-14)
        ip=root/f'random/ifa_task{t}.json'
        info=js(ip) if ip.exists() else None
        stages[t]=dict(epochs=100,batches=batches,lr_first=rr[0]['lr_first'],lr_final=rr[-1]['lr_after'],
            first_epoch_loss=rr[0]['loss'],last_epoch_loss=rr[-1]['loss'],
            selected_pairs=info['info']['selected_pairs'] if info else None,
            ifa_evidence='PRESENT' if info else 'NOT_IN_LOCAL_LIGHTWEIGHT_COPY')
    r={};record_hash_matches=0
    for i in range(6,10):
        for j in range(6,i+1):
            p=root/f'random/evaluation/trials_stage{i}_task{j}.jsonl';rows=lines(p)
            assert len(rows)==20 and sorted(x['initial_state_index'] for x in rows)==list(range(20))
            assert sorted(x['trial'] for x in rows)==list(range(20))
            for x in rows:
                assert x['seed']==seed and x['stage']==i and x['task']==j and x['epoch']==99
                assert x['loaded_downstream_keys']==100 and type(x['success']) is bool
                assert x['checkpoint_sha256']==complete['checkpoints'][f'mlrifa_{i}_model_ep99.pth']
                assert x['modal_sha256']==hashlib.sha256((root/'MODAL_CONTRACT.json').read_bytes()).hexdigest()
                assert x['initial_state_sha256']==x['initial_hashes'][x['initial_state_index']]
                assert 0<x['vector_group_steps']<=600
            attempts=lines(root/f'random/evaluation/attempts_stage{i}_task{j}.jsonl')
            assert [x['event'] for x in attempts]==['environment_construct_start','environment_construct_complete',
                'vector_attempt_start','vector_attempt_complete','vector_attempt_start','vector_attempt_complete','environment_closed']
            ends=[x for x in attempts if x['event']=='vector_attempt_complete']
            for g,a in enumerate(ends):
                assert a['group']==g and a['env_seed']==seed+g
                assert a['successes']==[x['success'] for x in rows[g*10:(g+1)*10]]
            r[i,j]=F(sum(x['success'] for x in rows),20)
            if not case.endswith('FIRST'):
                bound={str(Path(k)):v for k,v in baseline[suite]['source_sha256'].items()}
                assert bound[str(p)]==hashlib.sha256(p.read_bytes()).hexdigest();record_hash_matches+=1
    m=metrics(r);exact[case]=m
    if not case.endswith('FIRST'):
        old=next(x for x in baseline[suite]['runs'] if x['seed']==seed and x['method']=='MLR_IFA')
        assert all(F(old['metrics'][k]['fraction'])==v for k,v in m.items())
        assert all(old['matrix'][f'{i}_{j}']==float(v*100) for (i,j),v in r.items())
    gpu=js(root/'GPU_PREFLIGHT.json')
    pp=root/'PRETRAIN_METADATA.json'
    pretrain=js(pp) if pp.exists() else None
    features_path=root/'FEATURES.json'
    feature_summary=None
    if features_path.exists():
        features=js(features_path)
        assert all(len(v)==10 for v in features['demos'].values())
        feature_summary=dict(counts=features['counts'],demonstrations={k:len(v) for k,v in features['demos'].items()})
    results[case]=dict(root=rel,records=200,attempts=20,epochs=400,updates=complete['steps'],
        gpu=gpu.get('gpu'),torch=gpu.get('torch'),source_binding=bindings,
        checkpoint_origin=cfg['pretrain'].split('/')[-1],pretrain_sha256=modal['frontend_sha256'],
        pretrain_use_state=pretrain.get('pretrain_use_state') if pretrain else 'METADATA_NOT_IN_COPY',
        config_summary=dict(use_state=True,gamma=cfg['gamma'],batch=10,epochs_per_stage=100,lr=1e-4,warmup=500,checkpoint=99),
        features=feature_summary,
        baseline_bound_episode_files=record_hash_matches,stages=stages,metrics=display(m),
        matrix={f'{i}_{j}':float(v*100) for (i,j),v in r.items()})

summary={s:aggregate([exact[f'{s}_{seed}'] for seed in (41,42,43)]) for s in ('OBJECT','GOAL')}
task_summary={s:{str(j):dict(first_learned_mean=sum(results[f'{s}_{seed}']['matrix'][f'{j}_{j}'] for seed in (41,42,43))/3,
    final_mean=sum(results[f'{s}_{seed}']['matrix'][f'9_{j}'] for seed in (41,42,43))/3,
    across_observed_stages_mean=sum(results[f'{s}_{seed}']['matrix'][f'{i}_{j}'] for seed in (41,42,43) for i in range(j,10))/(3*(10-j))) for j in range(6,10)} for s in ('OBJECT','GOAL')}
historic={}
base='execution/20261008_E1_protective_selection/baseline_evidence/git_archive/reports/final_paper_text_faithful'
for view in ('primary_ep99','secondary_selected'):
    jj=js(f'{base}/{view}/ifa_three_seed_summary.json');mm=[]
    for s,v in jj['per_seed'].items():
        r={(i+6,j+6):F(round(v['r'][i][j]),100) for i in range(4) for j in range(i+1)}
        m=metrics(r);mm.append(m)
        assert all(abs(float(m[k]*100)-v[k.lower()])<.001 for k in ('FWT','NBT','AUC'))
    historic[view]=dict(summary=aggregate(mm),selected_epochs={s:v['selected_epochs'] for s,v in jj['per_seed'].items()},
        evidence_level='EXISTING_MATRIX_RECOMPUTATION_NOT_FRESH_EPISODE_OR_CHECKPOINT_AUDIT')
historic['secondary_minus_primary']={k:historic['secondary_selected']['summary'][k]['mean_percent']-historic['primary_ep99']['summary'][k]['mean_percent'] for k in ('FWT','NBT','AUC')}

# Verify selected release archive files against original git blob identities.
archive=W/'execution/20260927_223849/source_audit.tar.gz';raw(archive)
blobs=js('execution/20261013_coreset_object/BASE_BLOBS.json');release={}
names=['README.md','libero/configs/policy/complete_policy.yaml','libero/lifelong/algos/base.py',
       'libero/lifelong/algos/mlr.py','libero/lifelong/test.py','libero/lifelong/models/complete_policy.py']
with tarfile.open(archive) as tf:
    for name in names:
        b=tf.extractfile(name).read();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        assert blob==blobs[name],(name,'not pristine pinned release')
        release[name]=dict(git_blob=blob,sha256=hashlib.sha256(b).hexdigest())

for p in ['execution/20261013_coreset_object/ADJUSTED_PROTOCOL.md',
    'execution/20261019_ra_object_conditional/PROTOCOL.md',
    'execution/20261016_seed43_three_arm_repeat_v1/PROTOCOL.md',
    'execution/20261017_zr_coreset_extension/reporting_baseline_revision/REPORT.md',
    'execution/20261016_seed43_readonly_audit/completed_analysis/REPORT.md',
    'execution/20261022_align_object_extension_goal_conditional/analysis_OBJECT_GOAL_20261023/REPORT.md',
    base+'/FINAL_REPORT.md',base+'/PROTOCOL.json',
    'execution/20261013_coreset_object/source/libero/lifelong/datasets.py',
    'execution/20261013_coreset_object/source/coreset_exp/export.py',
    'execution/20261013_coreset_extension/seed43/source/coreset_exp/storage.py',
    'AGENTS.md']:
    raw(p)
for rel,h in sources.items():assert hashlib.sha256((W/rel).read_bytes()).hexdigest()==h
out=dict(status='PASS_READONLY_RECORDS_AND_PROVENANCE_AUDIT_CAUSAL_ATTRIBUTION_PARTIAL',
    observed=datetime.datetime.now().astimezone().isoformat(),scope='Six manuscript MLR+IFA baseline runs; original OBJECT43 auxiliary; historical checkpoint sensitivity',
    scientific_operations=dict(new_training=0,new_evaluation=0,new_forward=0,checkpoint_loaded=0,changed_results=0),
    main_records=1200,auxiliary_first43_records=200,main_epochs=2400,main_optimizer_updates=sum(exact_case['updates'] for case,exact_case in results.items() if not case.endswith('FIRST')),
    results=results,summary=summary,per_task=task_summary,
    repeat43_minus_first43={k:float((exact['OBJECT_43'][k]-exact['OBJECT_43_FIRST'][k])*100) for k in exact['OBJECT_43']},
    object_sensitivity_using_first43=aggregate([exact['OBJECT_41'],exact['OBJECT_42'],exact['OBJECT_43_FIRST']]),
    historical_checkpoint_sensitivity=historic,release_blob_verification=release,
    source_file_count=len(sources),sources_unchanged=True)
paper=Path('C:/Users/Lenovo/Zotero/storage/8WVD3WFZ/Yu 等 - Lifelong imitation learning with multimodal latent replay and incremental adjustment.pdf')
out['paper']=dict(title=paper.name,sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),
    inspected_print_pages=[6743,6744,6745],formula_and_table_visually_checked=True)
out['pinned_release_commit']='a721807f2f7fa7f9833dcc85c592d78e9accd00b'
HERE.mkdir(exist_ok=True,parents=True)
(HERE/'AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
(HERE/'SOURCE_HASHES.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
print(json.dumps(dict(status=out['status'],summary=summary,per_task=task_summary,historical=historic['secondary_minus_primary'],files=len(sources)),ensure_ascii=False,indent=2))
