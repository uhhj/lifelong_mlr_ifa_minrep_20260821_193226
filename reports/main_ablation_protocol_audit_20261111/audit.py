"""Audit manuscript rows from existing text/outcomes only: no ML or SSH imports."""
from pathlib import Path
from fractions import Fraction as F
import ast, collections, csv, datetime, hashlib, json, math, re, statistics, zipfile
import xml.etree.ElementTree as ET

R=Path(__file__).resolve().parent
W=R.parents[1]
E=W/'execution'
bindings={}
def raw(p):
    p=Path(p)
    b=p.read_bytes()
    key=str(p.relative_to(W)).replace('\\','/') if p.is_relative_to(W) else str(p)
    bindings[key]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    return b
def J(p):return json.loads(raw(p).decode('utf-8-sig'))
def T(p):return raw(p).decode('utf-8-sig')
def light(e,k):
    x=e['light'][k]
    assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'],k
    return json.loads(x['text'])
def lines(p):return [json.loads(x) for x in T(p).splitlines() if x]
def metrics(m):
    return dict(AUC=sum(sum(m[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,
        FWT=sum(m[t,t] for t in range(6,10))/4,
        NBT=sum(sum(m[t,t]-m[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
        OldFinal=sum(m[9,t] for t in range(6,9))/3,NewFinal=m[9,9],
        FinalAll=sum(m[9,t] for t in range(6,10))/4)
cases={}
identity={}
epoch_audits={}
def verify_epochs(ident,epochs,suite,stage_epochs=(100,100,100,100)):
    batches=[157,227,309,362] if suite=='OBJECT' else [100,139,185,337]
    assert len(epochs)==sum(stage_epochs)
    total=0;physical=0
    for stage,batch,ne in zip(range(6,10),batches,stage_epochs):
        rr=[x for x in epochs if int(x.get('stage',x.get('task',-1)))==stage]
        assert [x['epoch'] for x in rr]==list(range(ne)),ident
        for ep,x in enumerate(rr):
            total+=batch
            assert x['batches']==batch and x['steps']==total,(ident,stage,ep)
            def lr(step):return 1e-4*(step/500 if step<500 else max(0,(ne*batch-step)/(ne*batch-500)))
            assert math.isclose(x['lr_first'],lr(ep*batch),abs_tol=1e-14)
            assert math.isclose(x['lr_after'],lr((ep+1)*batch),abs_tol=1e-14)
            physical+=x.get('physical_optimizer_calls',batch)
    epoch_audits[ident]=dict(epochs=len(epochs),accepted_steps=total,physical_optimizer=physical,lr_checked_every_epoch=True,stage_epochs=list(stage_epochs))
def add(method,suite,seed,rows,sources,expected=None,training=None):
    cell=collections.defaultdict(list)
    for x in rows:cell[int(x['stage']),int(x['task'])].append(x)
    assert set(cell)=={(s,t) for s in range(6,10) for t in range(6,s+1)}
    assert all(len(v)==20 for v in cell.values())
    for v in cell.values():
        ids=[int(x.get('initial_index',x.get('initial_state_index',-1))) for x in v]
        assert sorted(ids)==list(range(20))
    m={k:F(sum(bool(x['success']) for x in v),20) for k,v in cell.items()}
    met={k:float(v*100) for k,v in metrics(m).items()}
    if expected:
        for k,v in expected.items():assert abs(met[k]-v)<1e-8,(method,suite,seed,k,met[k],v)
    ident='%s/%s/%s'%(method,suite,seed)
    assert ident not in cases
    cases[ident]=dict(method=method,suite=suite,seed=seed,episodes=len(rows),
        matrix_percent={'%s_%s'%k:float(v*100) for k,v in m.items()},metrics_percent=met,
        outcome_sources=[str(x).replace('\\','/') for x in sources],training=training,
        noise_hash_records=sum('noise_sha256' in x for x in rows))
    identity[ident]={(int(x['stage']),int(x['task']),int(x.get('initial_index',x.get('initial_state_index',-1)))):dict(initial_file=x.get('initial_file_sha256'),initial=x.get('initial_sha256',x.get('initial_state_sha256')),noise=x.get('noise_sha256'),env_seed=x.get('env_seed'),attempt=x.get('attempt')) for x in rows}

manuscript=Path(r'C:/Users/Lenovo/Desktop/论文/论文写作/DEBR_Complete_Manuscript_20Equations_Algorithm_20261110.docx')
b=raw(manuscript)
with zipfile.ZipFile(manuscript) as z:xml=z.read('word/document.xml')
n={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
root=ET.fromstring(xml)
tables=[]
for i,t in enumerate(root.findall('.//w:tbl',n)):
    rows=[[''.join(x.text or '' for x in c.findall('.//w:t',n)) for c in r.findall('w:tc',n)] for r in t.findall('w:tr',n)]
    tables.append(dict(index=i,rows=rows))
assert len(tables)==5

agg=J(E/'20261101_ablation_summary/DATA.json')
orig=J(E/'20261107_mlr_ifa_reproduction_audit/AUDIT.json')
align=J(E/'20261107_align_cqs_audit/AUDIT.json')
main=J(E/'20261107_main_comparison_table/TABLE_AUDIT.json')
prior_checks=[]
for name,meta in align['source_bindings'].items():
    p=W/name;b=raw(p)
    assert hashlib.sha256(b).hexdigest()==meta['sha256'],('Prior source mismatch',name)
    prior_checks.append(name)

expected={}
for r in agg['rows']:
    if r['study'] in ['baseline','current']:
        expected[r['method'],r['suite'],r['seed']]=r['metrics_percent']
for suite in ['OBJECT','GOAL']:
    old=J(E/('20261025_ccbp_v4_full/%s_THREE_SEED_BASELINES.json'%suite))
    group=collections.defaultdict(list)
    for name,digest in old['source_sha256'].items():
        p=Path(name)
        if not p.name.startswith('trials'):continue
        pos=p.as_posix()
        method='RA_ALIGN' if '/current/ra_coreset/' in pos else ('RA' if '/baseline/ra_coreset/' in pos else None)
        if method:
            b=raw(p);assert hashlib.sha256(b).hexdigest()==digest
            rr=[json.loads(x) for x in b.decode('utf-8-sig').splitlines() if x]
            assert len({x['seed'] for x in rr})==1
            group[method,int(rr[0]['seed'])].append((p,rr))
    for seed in [41,42,43]:
        m=orig['results']['%s_%d'%(suite,seed)]
        base=E/m['root']/'random/evaluation'
        modal=J(E/m['root']/'MODAL_CONTRACT.json')
        for name,source in m['source_binding'].items():
            assert hashlib.sha256(raw(W/source)).hexdigest()==modal['source_sha256']['coreset_exp/'+name]
        files=sorted(base.glob('trials_stage*_task*.jsonl'))
        assert len(files)==10
        rows=[x for p in files for x in lines(p)]
        add('MLR_IFA',suite,seed,rows,files,expected['MLR_IFA',suite,seed],dict(epochs=400,optimizer=m['updates'],collection=0,source_binding=m['source_binding'],config=m['config_summary'],hardware=m['gpu'],root=m['root']))
        verify_epochs('MLR_IFA/%s/%d'%(suite,seed),J(E/m['root']/'random/epochs.json'),suite)
        for method in ['RA','RA_ALIGN']:
            files,rr=zip(*group[method,seed]);assert len(files)==10
            add(method,suite,seed,[x for rows in rr for x in rows],files,expected[method,suite,seed],dict(epochs=400,optimizer=105500 if suite=='OBJECT' else 76100,collection=0))
            epfile=Path(files[0]).parent.parent/'epochs.json'
            if not epfile.exists() and method=='RA':
                if suite=='OBJECT':epfile=E/('20261019_ra_object_conditional/completion_20261020/download/evidence/OBJECT_RA_seed%d/ra_coreset/epochs.json'%seed)
                elif seed==41:epfile=E/'20261019_ra_coreset_goal_seed41/progress_20261019_222601/completion_ra/ra_coreset/epochs.json'
                elif seed==42:epfile=E/'20261018_ra_coreset_goal_seed42/final_evidence/ra_coreset/epochs.json'
                else:epfile=E/'20261019_ra_coreset_goal_extension/progress_20261019_160822/seed43/evidence/ra_coreset/epochs.json'
            verify_epochs('%s/%s/%d'%(method,suite,seed),J(epfile),suite)
            complete=epfile.parent/'TRAINING_COMPLETE.json'
            if complete.exists():
                cp=J(complete)
                for rows in rr:
                    for x in rows:assert x['checkpoint_sha256']==cp['checkpoints']['mlrifa_%s_model_ep99.pth'%x['stage']]

cbase=E/'20261027_c_goal42_43_object41_42_43/final_readonly_20261027_222119/EVIDENCE.json'
c=J(cbase)
c41base=E/'20261025_ccbp_v3_full/final_readonly_20261025_181226/EVIDENCE.json'
c41=J(c41base)
qbase=E/'20261031_full_q_vs_latest_s_two_suites/final_readonly_20261101_032229/EVIDENCE.json'
q=J(qbase)
ss={s:E/p/'EVIDENCE.json' for s,p in {
    'OBJECT':'20261028_support_gate_full_s_object/final_readonly_20261029_201235',
    'GOAL':'20261029_support_gate_full_s_goal/final_readonly_20261030_111546'}.items()}
se={s:J(p) for s,p in ss.items()}
for suite in ['OBJECT','GOAL']:
    for seed in [41,42,43]:
        case='%s_%d'%(suite,seed);sp='seed%d/'%seed
        ce,cp=(c41,'') if case=='GOAL_41' else (c,case+'/')
        cr=[x for x in light(ce,cp+'OUTCOMES.json') if x['arm']=='C']
        add('C_normal',suite,seed,cr,[str(c41base if case=='GOAL_41' else cbase)+'#light/'+cp+'OUTCOMES.json'],expected['C_normal',suite,seed],dict(epochs=400,optimizer=105500 if suite=='OBJECT' else 76100,collection=0))
        verify_epochs('C_normal/%s/%d'%(suite,seed),light(ce,cp+'EPOCHS_C.json'),suite)
        for method,ev,prefix,path,arm in [('Q',q,case+'/',qbase,'Q'),('S_latest',se[suite],sp,ss[suite],'S')]:
            rr=light(ev,prefix+'OUTCOMES.json')
            physical=(195300 if suite=='OBJECT' else 142200) if arm=='S' else (105500 if suite=='OBJECT' else 76100)
            add(method,suite,seed,rr,[str(path)+'#light/'+prefix+'OUTCOMES.json'],expected[method,suite,seed],dict(epochs=400,accepted=105500 if suite=='OBJECT' else 76100,optimizer=physical,collection=40))
            verify_epochs('%s/%s/%d'%(method,suite,seed),light(ev,prefix+'EPOCHS_'+arm+'.json'),suite)

stripped=E/'20261110_s_stripped_mlr_ifa_align'
newbase=stripped/'final_readonly_engineering02_20261111_064151/EVIDENCE.json'
new=J(newbase)
newanalysis=J(stripped/'FINAL_ANALYSIS.json')
for suite in ['OBJECT','GOAL']:
    for seed in [41,42,43]:
        for v in ['MLR','ALIGN']:
            case='%s_%s_%d'%(suite,v,seed)
            exp={k:x['percent'] for k,x in newanalysis['results'][case]['metrics'].items()}
            add('STRIPPED_'+v,suite,seed,light(new,case+'/OUTCOMES.json'),[str(newbase)+'#light/'+case+'/OUTCOMES.json'],exp,dict(epochs=400,optimizer=105500 if suite=='OBJECT' else 76100,collection=0))
            verify_epochs('STRIPPED_%s/%s/%d'%(v,suite,seed),light(new,case+'/EPOCHS_A.json'),suite)

# Adjacent selection, equal-optimizer-budget and lambda tables in the same draft.
kcenter=J(E/'20261109_kcenter_ra_matched_comparison/RESULTS.json')
for r in kcenter['runs']:
    if r['method']!='K-Center':continue
    rows=[x for p in r['sources'] for x in lines(W/p)]
    add('K_CENTER',r['suite'],r['seed'],rows,r['sources'],{k:v['percent'] for k,v in r['metrics'].items()},dict(epochs=400,optimizer=105500 if r['suite']=='OBJECT' else 76100,collection=0))
    verify_epochs('K_CENTER/%s/%d'%(r['suite'],r['seed']),J((W/r['sources'][0]).parent.parent/'epochs.json'),r['suite'])
budgetbase=E/'20261030_equal_optimizer_budget_cs_two_suites/final_readonly_20261031_065009/EVIDENCE.json'
budget=J(budgetbase)
for suite in ['OBJECT','GOAL']:
    for seed in [41,42,43]:
        case='%s_%d'%(suite,seed);ident='C_equal_budget/%s/%d'%(suite,seed)
        add('C_equal_budget',suite,seed,light(budget,case+'/OUTCOMES.json'),[str(budgetbase)+'#light/'+case+'/OUTCOMES.json'],expected['C_equal_budget',suite,seed],dict(epochs=700,optimizer=195300 if suite=='OBJECT' else 142200,collection=0,stage_epochs=[100,200,200,200],lr_horizon_extended=True))
        verify_epochs(ident,light(budget,case+'/EPOCHS_C.json'),suite,(100,200,200,200))
lambdas={
 'OBJECT':E/'20261101_lambda_r_object_sensitivity/final_acceptance_20261103_v1/final_readonly_20261103_124306/EVIDENCE.json',
 'GOAL':E/'20261103_lambda_r_goal_sensitivity/final_readonly_20261104_122613/EVIDENCE.json'}
for suite,path in lambdas.items():
    ev=J(path)
    for seed in [41,42,43]:
        for tag,weight in [('L005',.05),('L020',.2)]:
            case='%s_%d'%(tag,seed)
            add(tag,suite,seed,light(ev,case+'/OUTCOMES.json'),[str(path)+'#light/'+case+'/OUTCOMES.json'],training=dict(epochs=400,optimizer=195300 if suite=='OBJECT' else 142200,collection=40,lambda_R=weight))
            verify_epochs('%s/%s/%d'%(tag,suite,seed),light(ev,case+'/EPOCHS_S.json'),suite)

summary={}
for method in sorted({x['method'] for x in cases.values()}):
    summary[method]={}
    for suite in ['OBJECT','GOAL']:
        rows=[cases['%s/%s/%d'%(method,suite,s)] for s in [41,42,43]]
        summary[method][suite]={k:dict(mean=statistics.mean(x['metrics_percent'][k] for x in rows),sample_sd=statistics.stdev(x['metrics_percent'][k] for x in rows)) for k in rows[0]['metrics_percent']}

rowmap=[('MLR+IFA','MLR_IFA'),('+ RA-Coreset','RA'),('+ Execution Supervision','C_normal'),('+ Support-Weighted Retention','Q'),('DEBR (Ours)','S_latest')]
for row,(label,method) in zip(tables[2]['rows'][2:],rowmap):
    assert row[0]==label,(row[0],label)
    for si,suite in enumerate(['OBJECT','GOAL']):
        for ki,k in enumerate(['FWT','NBT','AUC']):
            numbers=[float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',row[1+si*3+ki])]
            st=summary[method][suite][k]
            assert abs(numbers[0]-st['mean'])<=.05000001
            assert abs(numbers[1]-st['sample_sd'])<=.05000001

def check_display(text,st):
    numbers=[float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',text)]
    assert len(numbers)==2 and abs(numbers[0]-st['mean'])<=.05000001 and abs(numbers[1]-st['sample_sd'])<=.05000001,(text,st)
for row,method in zip(tables[1]['rows'][2:],['MLR_IFA','K_CENTER','RA']):
    for si,suite in enumerate(['OBJECT','GOAL']):
        for ki,k in enumerate(['FWT','NBT','AUC']):check_display(row[1+3*si+ki],summary[method][suite][k])
for row,method in zip(tables[3]['rows'][2:],['C_normal','C_equal_budget','S_latest']):
    for si,suite in enumerate(['OBJECT','GOAL']):
        budget=int(row[1+4*si].replace(',',''))
        assert budget==cases['%s/%s/41'%(method,suite)]['training']['optimizer']
        for ki,k in enumerate(['FWT','NBT','AUC']):check_display(row[2+4*si+ki],summary[method][suite][k])
current_suite=None
for row in tables[4]['rows'][1:]:
    if row[0]:current_suite='OBJECT' if 'OBJECT' in row[0] else 'GOAL'
    method={.05:'L005',.1:'S_latest',.2:'L020'}[float(row[1])]
    for ki,k in enumerate(['FWT','NBT','AUC']):check_display(row[2+ki],summary[method][current_suite][k])
for row,source in zip(tables[0]['rows'][2:],main['rows']):
    if source['method'] in ['MLR + IFA','Ours']:
        method='MLR_IFA' if source['method']=='MLR + IFA' else 'S_latest'
        for si,suite in enumerate(['OBJECT','GOAL']):
            for ki,k in enumerate(['FWT','NBT','AUC']):check_display(row[1+3*si+ki],summary[method][suite][k])
    else:
        for cell,v in zip(row[1:],source['values']):check_display(cell,dict(mean=v['mean'],sample_sd=v['sd']))

family={'MLR_IFA':'A: historical global-RNG/full-history','RA':'A: historical global-RNG/full-history','RA_ALIGN':'A: historical global-RNG/full-history','C_normal':'B: cached/per-trajectory Normal','Q':'B: cached/per-trajectory Normal','S_latest':'B: cached/per-trajectory Normal','STRIPPED_MLR':'B: cached/per-trajectory Normal','STRIPPED_ALIGN':'B: cached/per-trajectory Normal','K_CENTER':'A: historical global-RNG/full-history','C_equal_budget':'B: cached/per-trajectory Normal','L005':'B: cached/per-trajectory Normal','L020':'B: cached/per-trajectory Normal'}
for x in cases.values():x['evaluation_family']=family[x['method']]

# Confirm rollout semantics for all six current C/Q/S and twelve new stripped cases.
eval_files=[]
for suite in ['OBJECT','GOAL']:
    for seed in [41,42,43]:
        cfile=E/'20261025_ccbp_v3_full/evaluate.py' if (suite,seed)==('GOAL',41) else E/'20261027_c_goal42_43_object41_42_43'/('%s_%d'%(suite,seed))/'evaluate.py'
        sfile=E/('20261028_support_gate_full_s_object' if suite=='OBJECT' else '20261029_support_gate_full_s_goal')/('seed%d'%seed)/'evaluate.py'
        qfile=E/'20261031_full_q_vs_latest_s_two_suites/engineering_startup_v2'/('%s_%d'%(suite,seed))/'evaluate.py'
        extra=[('C_equal_budget',E/'20261030_equal_optimizer_budget_cs_two_suites'/('%s_%d'%(suite,seed))/'evaluate.py')]
        ls=E/('20261101_lambda_r_object_sensitivity' if suite=='OBJECT' else '20261103_lambda_r_goal_sensitivity')
        extra += [(tag,ls/('%s_%d'%(tag,seed))/'evaluate.py') for tag in ['L005','L020']]
        for method,p in [('C_normal',cfile),('Q',qfile),('S_latest',sfile)]+[('STRIPPED_'+v,stripped/'cases'/('%s_%s_%d'%(suite,v,seed))/'evaluate.py') for v in ['MLR','ALIGN']]+extra:
            text=T(p)
            checks={key:token in text for key,token in dict(per_trajectory_numpy='np.random.RandomState(20261024+',zeros_after_success='act[done]=0',new_environment_per_group='for group in [0,1]:',cache='hist.append(encode(front,data))',argmax='k=l.argmax(-1)',normal_formula='mu[ix,k]+sd[ix,k]*',six_hundred='range(600)',five_zero='range(5)',close='finally:env.close()').items()}
            assert all(checks.values()),(str(p),checks)
            eval_files.append(dict(method=method,suite=suite,seed=seed,path=str(p.relative_to(W)).replace('\\','/'),checks=checks))

formal_pairs=[]
for a,b in [('RA','MLR_IFA'),('RA_ALIGN','RA'),('C_normal','Q'),('Q','S_latest'),('C_equal_budget','S_latest'),('STRIPPED_MLR','S_latest'),('STRIPPED_ALIGN','STRIPPED_MLR'),('L005','S_latest'),('L020','S_latest')]:
    for suite in ['OBJECT','GOAL']:
        for seed in [41,42,43]:
            left=identity['%s/%s/%d'%(a,suite,seed)];right=identity['%s/%s/%d'%(b,suite,seed)]
            assert left.keys()==right.keys() and len(left)==200
            assert all(left[k]['initial_file']==right[k]['initial_file'] and left[k]['initial']==right[k]['initial'] and left[k]['initial'] for k in left)
            noise_available=all(left[k]['noise'] and right[k]['noise'] for k in left)
            if noise_available:
                assert all(left[k]['noise']==right[k]['noise'] and left[k]['env_seed']==right[k]['env_seed'] and left[k]['attempt']==right[k]['attempt']==1 for k in left)
            formal_pairs.append(dict(left=a,right=b,suite=suite,seed=seed,initial_identity_pairs=200,noise_env_attempt_pairs=200 if noise_available else 0))

same_cp=[]
for a in align['cases']:
    suite,seed=a['case'].split('_');seed=int(seed)
    assert all(x['historical_ALIGN_equal'] for x in a['Task6_parity'].values())
    values={m:cases['%s/%s/%d'%(m,suite,seed)]['matrix_percent']['6_6'] for m in ['RA_ALIGN','C_normal','Q','S_latest']}
    for m,label in [('C_normal','C'),('Q','Q'),('S_latest','S')]:
        cases['%s/%s/%d'%(m,suite,seed)]['initialization']=a['initialization'][label]
        cases['%s/%s/%d'%(m,suite,seed)]['training_config']=a['training_config']
    same_cp.append(dict(suite=suite,seed=seed,actual_tensor_equal_record=True,success_percent=values,different_rollout_results=len(set(values.values()))>1))

historical_ast=[]
def func_ast(p,cls=None,fn=None):
    tree=ast.parse(T(p));body=tree.body
    if cls:body=next(x for x in body if isinstance(x,ast.ClassDef) and x.name==cls).body
    return ast.dump(next(x for x in body if isinstance(x,ast.FunctionDef) and x.name==fn),include_attributes=False)
for suite in ['OBJECT','GOAL']:
    for seed in [41,42,43]:
        source=E/('20261019_ra_object_conditional/source' if suite=='OBJECT' else ('20261019_ra_coreset_goal_seed41/ra/source' if seed==41 else ('20261018_ra_coreset_goal_seed42/source' if seed==42 else '20261019_ra_coreset_goal_extension/source')))/'coreset_exp'
        old=orig['results']['%s_%d'%(suite,seed)]['source_binding']
        eq=func_ast(W/old['train.py'],'PaperMLR','observe')==func_ast(source/'train.py','PaperMLR','observe')
        rollout=func_ast(W/old['eval_metric.py'],fn='evaluate_one_task_success')==func_ast(source/'eval_metric.py',fn='evaluate_one_task_success')
        assert eq and rollout
        historical_ast.append(dict(suite=suite,seed=seed,PaperMLR_observe_AST_equal=eq,rollout_function_AST_equal=rollout,ra_source=str(source.relative_to(W))))

aux=orig['results']['OBJECT_43_FIRST'];auxroot=E/aux['root']/'random/evaluation'
auxrows=[x for p in sorted(auxroot.glob('trials_stage*_task*.jsonl')) for x in lines(p)]
assert len(auxrows)==200
auxcells={(s,t):F(sum(bool(x['success']) for x in auxrows if (x['stage'],x['task'])==(s,t)),20) for s in range(6,10) for t in range(6,s+1)}
auxmetrics={k:float(v*100) for k,v in metrics(auxcells).items()}
repeat_sensitivity=dict(first43=auxmetrics,main_repeat43=cases['MLR_IFA/OBJECT/43']['metrics_percent'],main_auc_mean=summary['MLR_IFA']['OBJECT']['AUC']['mean'],first43_auc_mean=statistics.mean([cases['MLR_IFA/OBJECT/41']['metrics_percent']['AUC'],cases['MLR_IFA/OBJECT/42']['metrics_percent']['AUC'],auxmetrics['AUC']]),note='Existing first run and later repeat1, not a new scientific execution or a recommendation to select the higher score.')

comparisons=[
 dict(left='RA',right='MLR_IFA',same_evaluation_rules=True,paired_noise=False,same_production_optimizer=True,verdict='Historical same-rule method comparison; selection, induced IFA/order and hardware must remain distinguished. Not a paired-noise counterfactual.'),
 dict(left='RA_ALIGN',right='RA',same_evaluation_rules=True,paired_noise=False,same_production_optimizer=True,verdict='Historical joint BC-target/valid-window change; cannot split target alignment from masking.'),
 dict(left='C_normal',right='RA_ALIGN',same_evaluation_rules=False,paired_noise=False,same_production_optimizer=True,verdict='Not an isolated execution-supervision ablation: new update path and rollout family also changed.'),
 dict(left='C_normal',right='RA',same_evaluation_rules=False,paired_noise=False,same_production_optimizer=True,verdict='Current manuscript hides the RA_ALIGN intermediate: BC target, mask, execution supervision and evaluation change together.'),
 dict(left='Q',right='C_normal',same_evaluation_rules=True,paired_noise=True,same_production_optimizer=True,verdict='Compare the added retention bundle: success collection, own fixed teachers, support weighting and retention/routing. Not support-calibration alone; interactions differ.'),
 dict(left='S_latest',right='Q',same_evaluation_rules=True,paired_noise=True,same_production_optimizer=False,verdict='Full gate-enabled vs always-Q trajectories; different physical compute and own future caches. Not equal-compute gate-only causality.'),
 dict(left='S_latest',right='MLR_IFA',same_evaluation_rules=False,paired_noise=False,same_production_optimizer=False,verdict='Main table crosses rollout families and training/interaction budgets; not a strictly unified-protocol improvement estimate.'),
 dict(left='STRIPPED_ALIGN',right='STRIPPED_MLR',same_evaluation_rules=True,paired_noise=True,same_production_optimizer=True,verdict='New common stripped-S implementation isolates the joint BC target and valid-window handling.'),
 dict(left='S_latest',right='STRIPPED_MLR',same_evaluation_rules=True,paired_noise=True,same_production_optimizer=False,verdict='Same rollout implementation supports an overall implementation-level comparison, with RA/data, losses, training calls and interactions all differing; it cannot repair historical RA scores.'),
 dict(left='S_latest',right='STRIPPED_ALIGN',same_evaluation_rules=True,paired_noise=True,same_production_optimizer=False,verdict='Overall RA/execution/retention/gate bundle, not RA alone; full S was reused, with recorded hardware differences.'),
 dict(left='STRIPPED_MLR',right='MLR_IFA',same_evaluation_rules=False,paired_noise=False,same_production_optimizer=True,verdict='Training-entry and evaluation reimplementation; no isolated attribution of the score change to a single cause.'),
 dict(left='S_latest',right='C_equal_budget',same_evaluation_rules=True,paired_noise=True,same_production_optimizer=True,verdict='Matches physical production optimizer calls only. C uses 100/200/200/200 epochs and an extended decay horizon; S uses 100 each and 40 training acquisition episodes. Not equal FLOPs, wall time, interactions or optimization trajectory.'),
 dict(left='RA',right='K_CENTER',same_evaluation_rules=True,paired_noise=False,same_production_optimizer=True,verdict='Historical selection comparison, with the same qualifications as RA versus Random.'),
 dict(left='L005/L020',right='S_latest',same_evaluation_rules=True,paired_noise=True,same_production_optimizer=True,verdict='Fixed-grid lambda sensitivity with the same algorithm and nominal budget, but different devices/recovery histories and own later caches. Not a q-calibration removal or lambda=0 ablation.'),
]

sources_to_bind=[
 '20261019_ra_object_conditional/PROTOCOL.md','20261019_ra_object_conditional/PARENT_SOURCE_COMPARISON.json',
 '20261018_ra_coreset_goal_seed42/PROTOCOL.md','20261019_ra_coreset_goal_seed41/PROTOCOL.md',
 '20261019_ra_coreset_goal_extension/EXTENSION_PROTOCOL.md',
 '20261022_align_object_extension_goal_conditional/PROTOCOL.md',
 '20261024_ccbp/evaluate.py','20261024_ccbp_v2_full/evaluate.py',
 '20261025_ccbp_v3_full/PROTOCOL.md','20261027_c_goal42_43_object41_42_43/PROTOCOL.md',
 '20261028_support_gate_full_s_object/PROTOCOL.md','20261029_support_gate_full_s_goal/PROTOCOL.md',
 '20261030_equal_optimizer_budget_cs_two_suites/PROTOCOL.md','20261031_full_q_vs_latest_s_two_suites/PROTOCOL.md',
 '20261107_ablation_table/table_ablation.tex','20261107_ablation_table/TABLE_AUDIT.json',
 '20261109_manuscript_review/mlr_paper.txt',
 '20261110_s_stripped_mlr_ifa_align/PROTOCOL.md','20261110_s_stripped_mlr_ifa_align/IMPLEMENTATION_NOTES.md',
 '20261110_s_stripped_mlr_ifa_align/updates.py','20261110_s_stripped_mlr_ifa_align/case_runtime_template.py',
 '20261110_s_stripped_mlr_ifa_align/PROTOCOL_SCOPE_REVIEW.json',
]
for x in sources_to_bind:raw(E/x)
out=dict(status='PASS_EXISTING_OUTCOMES_AND_SOURCE_RECOMPUTATION_WITH_PROTOCOL_DIFFERENCES',
    observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    scope='Latest delivered 20-equation manuscript main, component, selection, optimizer-budget and sensitivity tables; removed ALIGN intermediate and newly completed stripped baselines.',
    manuscript=str(manuscript),tables=tables,cases=cases,summary=summary,epoch_audits=epoch_audits,
    verified_prior_source_files=len(prior_checks),verified_evaluation_files=eval_files,formal_pair_identity_checks=formal_pairs,
    same_task6_tensor_different_evaluation=same_cp,historical_MLR_RA_AST_checks=historical_ast,object43_repeat_sensitivity=repeat_sensitivity,comparisons=comparisons,
    main_external_rows=[dict(method=r['method'],source='MLR+IFA paper Table 1; literature numbers, not project reruns',values=r['values'],exact_training_and_rollout_in_project='NO_LOCAL_RUN_OR_PER_EPISODE_EVIDENCE') for r in main['rows'][:7]],
    source_bindings=bindings,new_training=0,new_forward=0,new_model_load=0,new_environment=0,remote_connections=0,
    limits=['Source and original accepted metadata/outcomes only; no current remote state or new CPU checkpoint acceptance.',
            'The existence of a frozen PROTOCOL does not establish user approval of each scientific deviation.',
            'Common rules permit performance comparison without exact common noise; exact paired SF/FS requires matched initial/noise evidence.',
            'Protocol mismatch does not imply fabricated outcomes, nor does correct outcome arithmetic establish protocol equivalence.'])
(R/'AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (R/'CASE_METRICS.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['method','suite','seed','evaluation_family','episodes','AUC','FWT','NBT','OldFinal','NewFinal','FinalAll'])
    writer.writeheader()
    for x in cases.values():writer.writerow({k:x[k] for k in ['method','suite','seed','evaluation_family','episodes']}|x['metrics_percent'])
(R/'MANUSCRIPT_TABLES.json').write_text(json.dumps(dict(manuscript=str(manuscript),tables=tables),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=out['status'],cases=len(cases),formal_records_recomputed=sum(x['episodes'] for x in cases.values()),prior_sources_verified=len(prior_checks),evaluation_sources_verified=len(eval_files),same_tensor_cases_with_different_success=sum(x['different_rollout_results'] for x in same_cp),new_training=0,new_environment=0),ensure_ascii=False))
