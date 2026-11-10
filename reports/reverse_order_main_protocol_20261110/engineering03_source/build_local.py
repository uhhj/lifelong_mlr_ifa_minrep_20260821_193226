"""Compose sources of the exact main-table methods; never execute science here."""
from pathlib import Path,PurePosixPath
import ast,copy,difflib,hashlib,json,re,sys
R=Path(__file__).resolve().parent
P=R.parent
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pre=J(Path(sys.argv[1])/'EVIDENCE.json')
binding=J(R/'MAIN_TABLE_BINDING.json')
audit=J(R/'PREFLIGHT_LOCAL_AUDIT.json')
assert sha(Path(sys.argv[1])/'EVIDENCE.json')==audit['sha256']
NAME='REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering03'
Q=PurePosixPath('/data/mlr_ifa/experiment_workspaces')/NAME
PRIVATE=PurePosixPath('/data/mlr_ifa/experiments')/NAME
ORDER=[0,1,2,3,4,5,9,8,7,6]
bundle={};review={};spec={}
def put(rel,text,source=None,changes=None):
    path=R/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(text.encode('utf-8'))
    bundle[rel]=text
    ast.parse(text,filename=rel,feature_version=(3,8)) if rel.endswith('.py') else None
    if source:
        old=Path(source).read_bytes().decode('utf-8-sig')
        review[rel]=dict(parent=str(source),parent_sha256=sha(source),new_sha256=sha(path),byte_identical=old.encode()==text.encode(),allowed_changes=changes or [])
        if old!=text:
            diff=''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile=str(source),tofile=rel))
            dest=R/'review'/(rel.replace('/','__')+'.diff');dest.parent.mkdir(exist_ok=True);dest.write_text(diff,encoding='utf-8')
def dump(rel,x):put(rel,json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def replace_function(text,name,new):
    nodes=ast.parse(text).body;node=next(x for x in nodes if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
    lines=text.splitlines(True)
    return ''.join(lines[:node.lineno-1])+new.rstrip()+'\n'+''.join(lines[node.end_lineno:])
for suite in ['OBJECT','GOAL']:
    for seed in [41,42,43]:
        original_case=suite+'_'+str(seed);raw_parent=pre['parents'][original_case]
        parent={k:v for k,v in raw_parent.items() if k!='tasks'};parent['tasks']={}
        for task,row in raw_parent['tasks'].items():
            keep=['current_count','baseline_feature_root','debr_feature_root','labels_path','labels_sha256','labels_binding_path','labels_binding_sha256','random_storage_path','ra_storage_path']
            parent['tasks'][task]={k:row[k] for k in keep}
            for method_key,old_key in [('random_selected','random_storage'),('ra_selected','ra_storage')]:
                parent['tasks'][task][method_key]=copy.deepcopy(row[old_key]['selected'])
        counts={i:parent['tasks'][str(ORDER[i])]['current_count'] for i in range(6,10)}
        selected={i:(counts[i]+1)//2 for i in counts}
        population={s:counts[s]+sum(selected[t] for t in range(6,s)) for s in range(6,10)}
        batches={s:(population[s]+9)//10 for s in population}
        budget=sum(batches.values())*100
        physical=(sum(batches.values())*2-batches[6])*100
        blocks=sum((batches[s]+9)//10 for s in range(7,10))*100
        starts=[sum(batches[s] for s in range(6,t))*100 for t in range(6,10)]
        assert (budget,physical,blocks)==((102100,191000,9100) if suite=='OBJECT' else (90700,161700,7200))
        for method in ['MLR','DEBR']:
            case=suite+'_'+method+'_'+str(seed);cr=Path('cases')/case
            crs=cr.as_posix();remote_case=Q/crs
            inp=dict(case=case,suite=suite,seed=seed,method=method,order=ORDER,order_index=21,original_case=original_case,
                     main_table_source=binding['cases'][original_case][method],accepted_budget=budget,physical_budget=budget if method=='MLR' else physical,
                     batches=batches,population=population,current_counts=counts,memory_counts=selected,gate_blocks=0 if method=='MLR' else blocks,
                     training_collection=0 if method=='MLR' else 40,formal=200,private_root=str(PRIVATE/case),parent=parent,
                     actual_stage_admission_required=True)
            if method=='MLR':
                source=Path(binding['cases'][original_case]['MLR']['source_root'])
                cfg=J(binding['cases'][original_case]['MLR']['config_path'])
                ec=J(binding['cases'][original_case]['MLR']['eval_config_path'])
                assert cfg['policy']==parent['main_mlr_config']['policy']
                assert cfg['train']==parent['main_mlr_config']['train']
                old_input=J(P/'20261110_s_stripped_mlr_ifa_align/cases'/(suite+'_MLR_'+str(seed))/'INPUT.json')
                inp['initial_state']=old_input['initial_state']
                inp['source_root']=str(remote_case/'source')
                for path in sorted(source.rglob('*')):
                    if not path.is_file() or path.suffix not in ['.py','.yaml','.yml'] or '__pycache__' in path.parts:continue
                    rel=path.relative_to(source).as_posix()
                    # Main-table provenance is per file, not merely per source root.
                    bound=binding['cases'][original_case]['MLR']['sources'].get(Path(rel).name) if rel.startswith('coreset_exp/') else None
                    if bound:
                        path=Path(bound['path']);assert sha(path)==bound['sha256']
                    text=path.read_bytes().decode('utf-8-sig')
                    changes=[]
                    if rel=='coreset_exp/common.py':
                        text,n=re.subn(r"assert SEED in \([^\n]+\), '[^'\n]+'", "assert SEED in (41,42,43), 'Only authorized reverse-order seeds'",text,count=1)
                        assert n==1 or (suite=='OBJECT' and seed==42 and 'assert SEED in' not in text)
                        changes=['authorize requested seeds41/42/43 in copied namespace helper only'] if n else []
                    if rel=='coreset_exp/train.py':
                        old_tree=ast.parse(text)
                        needle="get_benchmark('LIBERO_%s')(0)"%suite
                        assert text.count(needle)==1
                        text=text.replace(needle,"get_benchmark('LIBERO_%s')(c.data.task_order_index)"%suite)
                        nl='\r\n' if '\r\n' in text else '\n'
                        needle='a,b=build(c);history=[]';assert text.count(needle)==1
                        text=text.replace(needle,needle+nl+'        from reverse_admission import record_initial;record_initial(a)')
                        needle='            a.start_task(-1,loader)';assert text.count(needle)==1
                        text=text.replace(needle,needle+nl+'            from reverse_admission import admit;admit(a,dataset,t,info)')
                        needle="                history.append(row);write(root/'epochs.json',history)";assert text.count(needle)==1
                        text=text.replace(needle,"                row['sampler_sha256']=hashlib.sha256(np.asarray(list(iter(sampler)),dtype=np.int64).tobytes()).hexdigest()"+nl+needle)
                        needle='modes={n:m.training for n,m in a.policy.named_modules()},rng=cpu(rng()),step=a.step)';assert text.count(needle)==1
                        text=text.replace(needle,'modes={n:m.training for n,m in a.policy.named_modules()},requires={n:p.requires_grad for n,p in a.policy.named_parameters()},rng=cpu(rng()),step=a.step)')
                        # The actual main-table observe formula remains AST-identical.
                        old_class=next(x for x in old_tree.body if isinstance(x,ast.ClassDef) and x.name=='PaperMLR')
                        new_class=next(x for x in ast.parse(text).body if isinstance(x,ast.ClassDef) and x.name=='PaperMLR')
                        assert ast.dump(old_class,include_attributes=False)==ast.dump(new_class,include_attributes=False)
                        changes=['explicit reverse benchmark index','record actual official initialization','restored isolated own-stage admission before production','sampler identity and requires_grad metadata only']
                    put(crs+'/source/'+rel,text,path,changes)
                o=PRIVATE/case/'mlr_run';output=o/'random'
                for c in [cfg,ec]:
                    for field in ['pretrain','folder','bddl_folder','init_states_folder']:
                        c[field]=parent['main_mlr_config'][field]
                    c['data']['task_order_index']=21
                cfg.update(feature_dir=str(o/'features'),buffer_dir=str(output/'buffer'),experiment_dir=str(output/'models'),run_root=str(output),contract=str(o/'CONTRACT.json'))
                ec['feature_dir']=str(o/'features');ec['buffer_dir']=str(output/'buffer');ec['experiment_dir']=str(output/'evaluation')
                inp.update(input_root=str(o),output_root=str(output),train_config=cfg,eval_config=ec)
            else:
                source=Path(binding['cases'][original_case]['DEBR']['parent_directory'])
                cfg=copy.deepcopy(parent['debr_config']);ec=copy.deepcopy(parent['debr_eval_config'])
                o=PRIVATE/case/'inputs';output=PRIVATE/case/'S'
                for c in [cfg,ec]:c['data']['task_order_index']=21
                cfg.update(feature_dir=str(o/'features'),buffer_dir=str(o/'ra_coreset/buffer'),experiment_dir=str(output/'models'),run_root=str(output))
                inp.update(input_root=str(o),output_root=str(output),source_root=parent['debr_align_source'],train_config=cfg,eval_config=ec)
                existing=J(P/'20261110_s_stripped_mlr_ifa_align/FULL_S_BASELINE_BINDING.json')['cases'][original_case]
                w=Path(R.parents[1])/existing['initial_evidence_path']
                if not w.exists():w=Path(str(w).replace('\\','/'))
                init=J(w);inp['initial_state']={k:init[k] for k in ['tensors','rng']}
                names=['train.py','updates.py','legacy_updates.py','support_memory.py','fork_tools.py','gate_admission.py','legacy_losses.py','losses.py','memory.py','legacy_memory.py','initialization.py','collect.py','evaluate.py','synthetic_checks.py']
                for name in names:
                    path=source/name;assert path.exists(),path
                    text=path.read_bytes().decode('utf-8-sig');changes=[]
                    if name=='train.py':
                        old_budget=105500 if suite=='OBJECT' else 76100;old_physical=195300 if suite=='OBJECT' else 142200;old_blocks=9100 if suite=='OBJECT' else 6700
                        needle='assert a.step==%d and len(history)==400 and physical==%d and len(gates)==%d'%(old_budget,old_physical,old_blocks)
                        assert text.count(needle)==1; text=text.replace(needle,'assert a.step==%d and len(history)==400 and physical==%d and len(gates)==%d'%(budget,physical,blocks))
                        # Old Task6 comparison is not meaningful after changing which task comes first.
                        lo=text.index('            if stage==6:\n                baseline=') if '            if stage==6:\n                baseline=' in text else text.index('            if stage==6:\r\n                baseline=')
                        hi=text.index("            write(R/('STAGE_",lo)
                        text=text[:lo]+text[hi:];changes=['reverse population-derived terminal budgets','remove old-order Task6 diagnostic only']
                    if name in ['collect.py','evaluate.py']:
                        # Identity annotations only; normal sampling and environment steps stay unchanged.
                        text=text.replace('initial_index=i,','original_task=ORIGINAL_TASKS[taskid],initial_index=i,')
                        changes=['record original benchmark identity alongside learning position']
                    put(crs+('/collect_native.py' if name=='collect.py' else '/'+name),text,path,changes)
                    if name=='collect.py':
                        wrapper="import os,runpy\nrunpy.run_path(%r,run_name='__main__')\n"%str(Q/'collect_guard.py')
                        put(crs+'/collect.py',wrapper)
                candidate=P/'20261028_support_gate_full_s_object/engineering_copy_20261029_v1/candidate.py'
                put(crs+'/candidate.py',candidate.read_bytes().decode('utf-8-sig'),candidate)
                rt=(source/'runtime.py').read_text(encoding='utf-8')
                # Preserve its Dataset/build/setup/restore semantics, replacing only roots and integrity/capacity wrappers.
                rt=re.sub(r"^O=Path\([^\n]+\)","O=Path(%r)"%str(o),rt,count=1,flags=re.M)
                marker='sys.path.insert(0,str(S))';assert marker in rt
                rt=rt.replace(marker,marker+'\nfrom order import install,ORIGINAL_TASKS\ninstall()',1)
                rt=replace_function(rt,'verify',"def verify():\n from reverse_common import verify_case\n return verify_case(R)")
                rt=replace_function(rt,'capacity',"def capacity():\n from reverse_common import capacity as check\n return check()")
                # Replace inherited copy optimization guards with this order's actual stage starts.
                suffix=rt.find('\n# Independent snapshots:')
                if suffix>=0:rt=rt[:suffix]
                rt+='''\nfrom candidate import cpu_snapshot
_reference_cpu=cpu
_reference_capture=capture
_copy_checked={}
cpu=cpu_snapshot
def capture(a):
 result=_reference_capture(a)
 if a.step in STAGE_STARTS and a.scheduler.last_epoch==0 and str(a.step) not in _copy_checked:
  global cpu
  try:
   cpu=_reference_cpu;oracle=_reference_capture(a)
  finally:cpu=cpu_snapshot
  assert equal(result,oracle),'New order actual full state copy mismatch'
  _copy_checked[str(a.step)]=dict(all_state_exact=True,optimizer_calls_added=0,environments_added=0)
  write(R/'COPY_EQ.json',dict(status='PASS_OBSERVED_STAGE_STARTS',checked=_copy_checked,candidate_sha256=sha(R/'candidate.py')))
 return result
'''
                put(crs+'/runtime.py',rt,source/'runtime.py',['new input/output roots','explicit order installation','new frozen integrity and global capacity guard','actual-stage full-state CPU-copy oracle'])
                layout="""from pathlib import Path
import json
R=Path(__file__).resolve().parent
ARMS='S'
SEED=%d
NAME=%r
ROOT_PRIVATE=Path(%r)/'unused_root_payload'
DATA_PRIVATE=Path(%r)
OUTPUT={'S':Path(%r)}
VOLUME_BOUND={'/root':0,'/data':14*1024**3}
_counts=json.loads((R/'INPUT_COUNTS.json').read_text())
POP={int(k):v for k,v in _counts['population'].items()}
VALID={int(k):v for k,v in _counts['valid'].items()}
OLD={int(k):v for k,v in _counts['old_valid'].items()}
STAGE_STARTS=%r
MEMORY_PER_TASK=128
ANCHOR_BATCH=4
RETAIN_WEIGHT=.1
def model_path(arm,stage):return OUTPUT[arm]/('stage%%d.pth'%%stage)
def memory_path(arm,task):return OUTPUT[arm]/'memory'/('task%%d.pth'%%task)
"""%(seed,NAME,str(Q),str(PRIVATE),str(output),starts)
                put(crs+'/layout.py',layout)
            spec[case]=inp
            dump(crs+'/INPUT.json',inp)
for name in ['order.py','instrumentation.py','reverse_admission.py','reverse_common.py','run_case.py','input_prepare_case.py','collect_guard.py','controller.py','final_cpu_case.py','final_readonly.py','check_readonly.py']:
    path=R/name
    if path.exists():put(name,path.read_text(encoding='utf-8'))
dump('SPEC.json',dict(name=NAME,root=str(Q),private_root=str(PRIVATE),order=ORDER,cases={k:dict(input_file='cases/'+k+'/INPUT.json',suite=v['suite'],method=v['method'],seed=v['seed'],accepted_budget=v['accepted_budget'],physical_budget=v['physical_budget'],gate_blocks=v['gate_blocks'],training_collection=v['training_collection'],formal=v['formal']) for k,v in spec.items()}))
dump('SOURCE_REVIEW.json',dict(status='STATIC_COMPOSITION_NOT_REMOTE_IMPORT_OR_GPU_PASS',files=review,order=ORDER,baseline_observe_AST_unchanged=True,trainings_started=0,environments_started=0))
(R/'DEPLOY_TEXT_BUNDLE.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(R/'UPLOAD_REVIEW.json').write_text(json.dumps(dict(status='TEXT_BUNDLE_PREPARED_NOT_DISPATCHED',files={k:dict(bytes=len(v.encode()),sha256=hashlib.sha256(v.encode()).hexdigest()) for k,v in bundle.items()},private_payloads_included=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='COMPOSED_NOT_DEPLOYED',cases=len(spec),files=len(bundle),epochs=4800,accepted=1156800,physical=1636500,formal=2400,training_collection=240)))
