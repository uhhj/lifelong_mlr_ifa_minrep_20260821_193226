"""Local, metadata-only audit of historical ALIGN and accepted C/Q/S lineage."""
from pathlib import Path
import ast
import hashlib
import json

R=Path(__file__).resolve().parent
W=R.parents[1]
E=W/'execution'
sources={}
def raw(p):
    p=Path(p);b=p.read_bytes();sources[p.relative_to(W).as_posix()]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest());return b
def js(p):return json.loads(raw(p).decode('utf-8-sig'))
def txt(p):return raw(p).decode('utf-8-sig').replace('\r\n','\n')
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def light(d,k):return json.loads(d['light'][k]['text'])
def historical(suite,seed):
    if (suite,seed)==('OBJECT',41):return E/'20261022_caet_v3_tac_eftr_object41/final_ALIGN_seed41_20261022_220252'
    stamps={'OBJECT':{42:'012252',43:'032427'},'GOAL':{41:'052225',42:'072244',43:'092244'}}
    return E/f'20261022_align_object_extension_goal_conditional/final_{suite}_{seed}_20261023_{stamps[suite][seed]}'

dirs=dict(C=E/'20261027_c_goal42_43_object41_42_43/final_readonly_20261027_222119',
          C41=E/'20261025_ccbp_v3_full/final_readonly_20261025_181226',
          Q=E/'20261031_full_q_vs_latest_s_two_suites/final_readonly_20261101_032229',
          S_GOAL=E/'20261029_support_gate_full_s_goal/final_readonly_20261030_111546',
          S_OBJECT=E/'20261028_support_gate_full_s_object/final_readonly_20261029_201235')
ev={};accepted={}
for name,p in dirs.items():
    ev[name]=js(p/'EVIDENCE.json');a=js(p/'LOCAL_FINAL_AUDIT.json')
    assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE',name
    bindings=[a[k] for k in ('evidence_sha256','source_sha256') if k in a]
    assert sha(p/'EVIDENCE.json') in bindings,name
    accepted[name]=dict(status=a['status'],evidence_sha256=sha(p/'EVIDENCE.json'))

bundles={
 'Q':js(E/'20261031_full_q_vs_latest_s_two_suites/engineering_startup_v2/DEPLOY_TEXT_BUNDLE.json'),
 'S_GOAL':js(E/'20261029_support_gate_full_s_goal/DEPLOY_TEXT_BUNDLE.json'),
 'S_OBJECT':js(E/'20261028_support_gate_full_s_object/DEPLOY_TEXT_BUNDLE.json')}
records=[]
for suite in ['OBJECT','GOAL']:
 for seed in [41,42,43]:
    case=f'{suite}_{seed}';h=historical(suite,seed);cur=h/'current'
    ha=js(h/'LOCAL_FINAL_AUDIT.json');assert ha['status']=='COMPLETED_LOCAL_INDEPENDENT_PASS'
    conf=js(cur/'resolved_ra_coreset.json');contract=js(cur/'CONTRACT.json')
    oldinit=js(cur/'ra_coreset/INITIAL_STATE.json');hist_epochs=js(cur/'ra_coreset/epochs.json')
    c=ev['C41'] if case=='GOAL_41' else ev['C'];cp='' if case=='GOAL_41' else case+'/'
    q=ev['Q'];qp=case+'/';s=ev['S_'+suite];sp=f'seed{seed}/'
    latest={a:light(d,p+'INITIAL_'+a+'.json') for a,d,p in [('C',c,cp),('Q',q,qp),('S',s,sp)]}
    init={a:dict(tensors_equal_historical=x['tensors']==oldinit['tensors'],
                 historical_cuda_rng_equal=x.get('historical_cuda_rng_equal'),
                 official_cp_sha256=x['official_cp_sha256'],hardware=x.get('hardware')) for a,x in latest.items()}
    assert all(v['tensors_equal_historical'] for v in init.values())
    assert len({v['official_cp_sha256'] for v in init.values()})==1
    stage=[]
    ce=light(c,cp+'EPOCHS_C.json');qe=light(q,qp+'EPOCHS_Q.json');se=light(s,sp+'EPOCHS_S.json')
    assert all(len(v)==400 for v in [hist_epochs,ce,qe,se])
    for t in range(6,10):
        mp=cur/f'ra_coreset/dataset_manifests/task{t}.json';man=js(mp)
        qi=light(q,qp+f'TRAIN_INPUT_Q_{t}.json');si=light(s,sp+f'TRAIN_INPUT_S_{t}.json')
        assert qi['manifest_sha256']==si['manifest_sha256']==sha(mp)
        if case!='GOAL_41':
            parent=js(E/'20261027_c_goal42_43_object41_42_43'/case/'PARENT_BINDING.json')
            assert parent['stages'][str(t)]['manifest_sha256']==sha(mp)
            assert parent['parent_contract_sha256']==sha(cur/'CONTRACT.json')
            c_manifest_proof='parent binding hash and historical frozen contract'
        else:
            import re
            c_runtime=txt(E/'20261025_ccbp_v3_full/engineering01_source/runtime.py')
            q_runtime=txt(E/'20261031_full_q_vs_latest_s_two_suites/engineering_startup_v2'/case/'runtime.py')
            assert re.search(r"O=Path\('([^']+)'\)",c_runtime).group(1)==re.search(r"O=Path\('([^']+)'\)",q_runtime).group(1)
            assert "dataset_manifests/task%d.json" in c_runtime
            c_manifest_proof='same historical O path and manifest-loading source; no separate manifest hash in C41 final light'
        he=[x for x in hist_epochs if x['task']==t];crows=[x for x in ce if x['stage']==t];qrows=[x for x in qe if x['stage']==t];srows=[x for x in se if x['stage']==t]
        assert all(len(v)==100 for v in [he,crows,qrows,srows])
        sched=lambda a:[(x['epoch'],x['batches'],x['lr_first'],x['lr_after']) for x in a]
        assert sched(he)==sched(crows)==sched(qrows)==sched(srows),(case,t,'schedule')
        assert [x['sampler_sha256'] for x in crows]==[x['sampler_sha256'] for x in qrows]==[x['sampler_sha256'] for x in srows]
        stage.append(dict(stage=t,population=len(man['rows']),valid=sum(bool(x['bc_valid']) for x in man['rows']),
                          manifest_sha256=sha(mp),historical_Q_S_manifest_hash_equal=True,C_manifest_proof=c_manifest_proof,
                          actual_epoch_batch_lr_equal=True,C_Q_S_sampler_equal=True))
    # Check all formal initial-state identities against the historical records.
    cr=light(c,cp+'OUTCOMES.json');cr=[x for x in cr if x['arm']=='C']
    qr=light(q,qp+'OUTCOMES.json');sr=light(s,sp+'OUTCOMES.json')
    idx=lambda rows:{(x['stage'],x['task'],x['initial_index']):x for x in rows}
    ci,qi,si=map(idx,[cr,qr,sr]);assert len(ci)==len(qi)==len(si)==200
    checked=0
    for t in range(6,10):
      for task in range(6,t+1):
        rows=[json.loads(x) for x in txt(cur/f'ra_coreset/evaluation/trials_stage{t}_task{task}.jsonl').splitlines() if x]
        assert len(rows)==20
        for x in rows:
          k=(t,task,x['initial_state_index']);old=(x['initial_file_sha256'],x['initial_state_sha256'])
          for i in [ci,qi,si]:assert old==(i[k]['initial_file_sha256'],i[k]['initial_sha256'])
          for field in ['noise_sha256','env_seed','attempt']:
            assert ci[k][field]==qi[k][field]==si[k][field],(case,k,field)
          checked+=1
    qsrc=E/'20261031_full_q_vs_latest_s_two_suites/engineering_startup_v2'/case
    ssrc=E/('20261029_support_gate_full_s_goal' if suite=='GOAL' else '20261028_support_gate_full_s_object')/f'seed{seed}'
    shared={f:txt(qsrc/f)==txt(ssrc/f) for f in ['updates.py','losses.py','support_memory.py','memory.py','fork_tools.py','legacy_losses.py']}
    assert all(shared.values()),(case,shared)
    for f in shared:
        assert txt(qsrc/f)==bundles['Q'][case+'/'+f].replace('\r\n','\n')
        assert txt(ssrc/f)==bundles['S_'+suite][f'seed{seed}/'+f].replace('\r\n','\n')
    records.append(dict(case=case,initialization=init,stages=stage,
        training_config=dict(gamma=conf['gamma'],pretrain=conf['pretrain'],train=conf['train']),
        Task6_parity={a:light(d,p+f'TASK6_PARITY_{a}.json') for a,d,p in [('C',c,cp),('Q',q,qp),('S',s,sp)]},
        Q_S_common_scientific_sources_LF_equal=shared,Q_S_source_matches_reviewed_deploy_bundles=True,formal_initial_records_checked=checked,
        current_C_Q_S_noise_env_attempt_pairing=True,historical_noise_pairing=False))

# Read exact objective and evaluator sources, retain line-numbered excerpts.
targets={
 'historical_ALIGN':'20261022_align_object_extension_goal_conditional/additions/caet_v3_exp/observe.py',
 'masked_BC':'20261022_align_object_extension_goal_conditional/additions/caet_v3_exp/model.py',
 'C_update':'20261027_c_goal42_43_object41_42_43/GOAL_42/updates.py',
 'C_aux':'20261027_c_goal42_43_object41_42_43/GOAL_42/losses.py',
 'Q_S_update':'20261031_full_q_vs_latest_s_two_suites/engineering_startup_v2/GOAL_41/updates.py',
 'Q_S_retention':'20261031_full_q_vs_latest_s_two_suites/engineering_startup_v2/GOAL_41/support_memory.py',
 'gate':'20261029_support_gate_full_s_goal/seed41/fork_tools.py',
 'evaluation':'20261029_support_gate_full_s_goal/seed41/evaluate.py',
 'historical_protocol':'20261022_align_object_extension_goal_conditional/PROTOCOL.md',
 'current_C_protocol':'20261027_c_goal42_43_object41_42_43/PROTOCOL.md',
 'Q_protocol':'20261031_full_q_vs_latest_s_two_suites/PROTOCOL.md'}
needles=['native=','VARIANT==\'ALIGN\'','loss=native','loss=bc','def masked_bc','target=','if arm == \'C\'','score = c','value =','retain_weight','def weights','support','factor=','def choose','checks=dict','standard_normal','control_seed(SEED);env','global动作RNG','gamma=']
excerpts={}
for label,rel in targets.items():
    st=txt(E/rel);excerpts[label]=dict(path=rel,lines=[dict(line=i,text=x.strip()) for i,x in enumerate(st.splitlines(),1) if any(n in x for n in needles)])
summary=js(E/'20261101_ablation_summary/DATA.json')
auc={suite:{method:summary['summary'][f'{study}/{suite}/{method}']['AUC']['mean'] for study,method in [('baseline','RA'),('baseline','RA_ALIGN'),('current','C_normal'),('current','Q'),('current','S_latest')]} for suite in ['OBJECT','GOAL']}
out=dict(status='PASS_LOCAL_LINEAGE_AND_PROTOCOL_AUDIT',cases=records,accepted_evidence=accepted,AUC_reference=auc,
         excerpts=excerpts,source_bindings=sources,new_training=0,new_environment=0,new_model_forward=0,
         verdict='Historical ALIGN is not latest C: C adds old-window execution expert cost and margin routing. A four-row strict toggle table omitting this component is invalid. Use an explicit expert-auxiliary C row/column and distinguish historical vs current evaluation protocols.')
(R/'AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],cases=len(records),stages=sum(len(x['stages']) for x in records),formal_initial_records=sum(x['formal_initial_records_checked'] for x in records),sources=len(sources)),indent=2))
