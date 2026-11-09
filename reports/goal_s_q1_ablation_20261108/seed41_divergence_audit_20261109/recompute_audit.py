"""Recompute seed41 divergence from accepted scalar evidence and CPU inspection.

No training, inference, rollout or remote commands. Outputs contain only
metadata, hashes, scalar comparisons and recorded outcome rates.
"""
from pathlib import Path
import ast,hashlib,json,struct
A=Path(__file__).resolve().parent;R=A.parent;B=R.parent/'20261029_support_gate_full_s_goal'
NP=R/'final_readonly_20261109_120106/EVIDENCE.json';BP=B/'final_readonly_20261030_111546/EVIDENCE.json'
CP=A/'capacity_readonly_20261109_121742/EVIDENCE.json'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
n,b,cpu=read(NP),read(BP),read(CP)
def light(d,case,name):
    v=d['light'][case+'/'+name]
    assert hashlib.sha256(v['text'].encode()).hexdigest()==v['sha256']
    return [json.loads(x) for x in v['text'].splitlines()] if name.endswith('.jsonl') else json.loads(v['text'])
def pair(name):return light(n,'Q1_41',name),light(b,'seed41',name)
assert cpu['status']=='PASS_CPU_READONLY_PAIR_COMPARISON' and cpu['files_unchanged_after']
assert all(cpu[k]==0 for k in ['optimizer_calls','environment_calls','model_forward_calls'])
assert cpu['files']==read(A/'CPU_AUDIT_BINDING.json')['expected']
out=dict(status='PASS_READONLY_DIVERGENCE_LOCAL_RECOMPUTATION',scope='GOAL seed41 accepted S versus fixed-effective-q1 S',sources={str(p.relative_to(R.parent)):sha(p) for p in [NP,BP,CP]},optimizer_calls=0,environment_calls=0,model_forward_calls=0)
pn,pb=read(R/'Q1_41/PARENT_BINDING.json'),read(B/'seed41/PARENT_BINDING.json')
out['parent_binding']={k:dict(equal=pn[k]==pb[k],entries=len(pn[k]) if isinstance(pn[k],(dict,list)) else None) for k in pb}
assert all(v['equal'] for v in out['parent_binding'].values())
ini,ibi=pair('INITIAL_S.json');out['hardware']={'q1':ini['hardware'],'s':ibi['hardware']}
assert ini['tensors']==ibi['tensors'] and ini['rng']==ibi['rng']
out['initial_model_rng_exact']=True
out['code']={}
for p in sorted((B/'seed41').glob('*.py')):
    q=R/'Q1_41'/p.name
    if not q.exists():continue
    st,qt=p.read_text(encoding='utf8'),q.read_text(encoding='utf8')
    out['code'][p.name]=dict(normalized_text_equal=st==qt,ast_equal=ast.dump(ast.parse(st))==ast.dump(ast.parse(qt)),s_sha256=sha(p),q1_sha256=sha(q))
out['wrapper_sources']={p.name:sha(p) for p in [R/'train_wrapper.py',R/'environment_locked.py']}
out['shared_source_inspection']={}
for name,local in [('libero/lifelong/models/brief.py','20261024_ccbp/reference_source/libero/lifelong/models/brief.py'),('libero/lifelong/models/policy_head.py','20261024_ccbp/reference_source/libero/lifelong/models/policy_head.py'),('coreset_exp/model.py','20261022_align_object_extension_goal_conditional/additions/coreset_exp/model.py')]:
    p=R.parent/local;raw=p.read_bytes();lf=raw.replace(b'\r\n',b'\n');variants=[raw,lf,lf.replace(b'\n',b'\r\n')]
    assert pn['source_sha256'][name] in [hashlib.sha256(x).hexdigest() for x in variants]
    out['shared_source_inspection'][name]=dict(local=str(p),local_sha256=sha(p),bound_sha256=pn['source_sha256'][name],text_matches_bound_with_line_ending_normalization=True)
en,eb=pair('EPOCHS_S.json');assert len(en)==len(eb)==400
out['stages']={};out['gates']={}
on,ob=pair('OUTCOMES.json')
for t in range(6,10):
    ee=[x for x in en if x['stage']==t];ff=[x for x in eb if x['stage']==t];assert len(ee)==len(ff)==100
    epoch={k:next((i for i,(x,y) in enumerate(zip(ee,ff)) if x[k]!=y[k]),None) for k in ['sampler_sha256','lr_first','lr_after','steps','batches','totals','training_probe','gate_Q_accepts']}
    assert all(epoch[k] is None for k in ['sampler_sha256','lr_first','lr_after','steps','batches'])
    cn,cb=pair('COLLECTION_OUTCOMES_S_%d.json'%t);assert len(cn)==len(cb)==10
    keys=['initial_index','initial_sha256','initial_file_sha256','noise_sha256','env_seed','attempt']
    assert all(all(x[k]==y[k] for k in keys) for x,y in zip(cn,cb))
    mn,mb=pair('MEMORY_S_%d.json'%t);cm=cpu['stages'][str(t)]
    maps=[{(x['trajectory'],x['step']):x for x in z['rows']} for z in [mn,mb]];common=sorted(maps[0].keys()&maps[1].keys())
    assert len(common)==cm['common_rows']
    exact={k:sum(maps[0][i][k+'_sha256']==maps[1][i][k+'_sha256'] for i in common) for k in ['feature','mu','sd']}
    assert all(exact[k]==sum(row[k]['exact'] for row in cm['rows']) for k in exact)
    outcomes=[]
    for j in range(6,t+1):
        aa=[x for x in on if x['stage']==t and x['task']==j];bb=[x for x in ob if x['stage']==t and x['task']==j]
        assert len(aa)==len(bb)==20
        assert all(all(x[k]==y[k] for k in keys) for x,y in zip(aa,bb))
        outcomes.append(dict(task=j,q1=100*sum(x['success'] for x in aa)/20,s=100*sum(x['success'] for x in bb)/20))
    out['stages'][str(t)]=dict(first_epoch_difference_zero_based=epoch,checkpoint_different_tensors=cm['checkpoint_different_count'],checkpoint_max_abs=max(x['max_abs'] for x in cm['checkpoint_tensors'].values()),collection_successes=dict(q1=sum(x['success'] for x in cn),s=sum(x['success'] for x in cb)),action_streams_different=sum(x['action_stream_sha256']!=y['action_stream_sha256'] for x,y in zip(cn,cb)),collection_steps=[dict(index=x['initial_index'],q1=x['steps'],s=y['steps']) for x,y in zip(cn,cb)],memory_counts=cm['cache_rows'],memory_common=len(common),memory_common_exact=exact,normalization=cm['scales'],formal_cells=outcomes)
    if t>6:
        g,h=pair('GATES_stage%d.jsonl'%t);assert len(g)==len(h)
        idx=next(i for i,(x,y) in enumerate(zip(g,h)) if x['accepted']!=y['accepted'])
        first_c=next((i for i,(x,y) in enumerate(zip(g,h)) if any(x['C'][k]!=y['C'][k] for k in ['bc_new','exec_new'])),None)
        pr,ps=pair('PROBES_%d.json'%t)
        assert pr['support']==ps['support']=={str(j):1. for j in range(6,t)}
        out['gates'][str(t)]=dict(total=len(g),different_decisions=sum(x['accepted']!=y['accepted'] for x,y in zip(g,h)),first_different_decision_zero_based=idx,first_C_new_score_difference_zero_based=first_c,first_different_decision=dict(q1=g[idx],s=h[idx]),initial_block=dict(q1=g[0],s=h[0]),first_eight_choices_q1=[x['accepted'] for x in g[:8]],first_eight_choices_s=[x['accepted'] for x in h[:8]],effective_q=pr['support'],q1_empirical_support=pr['empirical_support'],probe_new_indices_equal=pr['new_indices']==ps['new_indices'],probe_old_indices_equal=pr['old_indices']==ps['old_indices'])
cm=cpu['stages']['6'];frames={(x['trajectory'],x['absolute_step']):x for x in cm['task6_saved_frames']}
out['online_first_saved_witnesses']=sorted([r for r in cm['rows'] if r['feature']['exact'] and not(r['mu']['exact'] and r['sd']['exact'])],key=lambda r:(r['step'],r['trajectory']))
assert [(r['initial_index'],r['step']) for r in out['online_first_saved_witnesses']]==[(21,6),(23,6)]
out['earliest_different_saved_feature']=min((x for x in frames.values() if not x['exact']),key=lambda x:x['absolute_step'])
out['saved_frame_count']=len(frames)
out['all_available_frames_through_step5_exact']=all(x['exact'] for x in frames.values() if x['absolute_step']<=5)
out['sampling_limit']='Reservoir windows are incomplete; first saved witness is not proven first arithmetic/action mismatch across all trajectories.'
out['total_different_gate_decisions']=sum(x['different_decisions'] for x in out['gates'].values())
out['driver_claim']='No contemporaneous per-kernel/driver trace; numerical backend/hardware cause strongly supported but exact CUDA operator/driver failure not identified.'
(A/'AUDIT_SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],first_saved_output_difference='Task6 collection step6 (7th policy action), initial indices21/23, identical input tensors',first_gate_flip='Task7 epoch0 batch70 (8th block)',gate_decisions_different=out['total_different_gate_decisions'],cost='0 optimizer / 0 environment / 0 forward',file='AUDIT_SUMMARY.json'),ensure_ascii=False))
