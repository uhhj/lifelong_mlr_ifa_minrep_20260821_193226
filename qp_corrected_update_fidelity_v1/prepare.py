import collections,copy,hashlib,json,random,subprocess,sys,shutil,time
import h5py,numpy as np,torch
from .common import *
def main():
 assert not (OUT/'PROTOCOL.json').exists()
 from cb_ifa_experiment import gpu_runtime as g
 from libero.libero.benchmark import get_benchmark
 from sample_qp_lifelong_v1_continuation_v4.common import selected_references
 from sample_qp_optimized_v1_continuation_v4 import solver
 source=read(SOURCE/'START_MANIFEST.json');starts=source['starts'];prior=read(SOURCE/'PROBE_MANIFEST.json')['windows']
 refs_all=[];old_all=[];new_all=[];changes=[];selection={};lineage={};ambiguity=[]
 for seed in (45,46,47):
  cfg=g.train_config(seed,SCRATCH/'bootstrap');bench=get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index);indices={}
  for task in (6,7,8):
   idx={}
   with h5py.File(ROOT/'data'/bench.get_task_demonstration(task),'r') as f:
    for demo in sorted(f['data'],key=lambda x:int(x.split('_')[-1]))[:10]:
     actions=np.asarray(f['data'][demo]['actions'],np.float32)
     for t in range(len(actions)):
      key=np.ascontiguousarray(actions[np.minimum(np.arange(t,t+8),len(actions)-1)]).tobytes()
      idx.setdefault(key,[]).append([demo,t,len(actions)])
   indices[task]=idx
  for task in (7,8):
   group=[s for s in starts if s['seed']==seed and s['task_id']==task];plan=read(ROOT/group[0]['plan']['path']);records=[];pool=[]
   for i,(origin,h) in enumerate(zip(plan['immutable_feature_origins'],plan['record_sha256'])):
    t=int(Path(origin).parent.name);records.append(dict(origin=origin,task=t,sha256=h))
    if origin not in lineage:
     z=torch.load(ROOT/origin,map_location='cpu');key=np.ascontiguousarray(z['actions'].numpy().reshape(8,7),np.float32).tobytes();matches=indices[t].get(key,[]);assert matches
     lineage[origin]=matches
    matches=lineage[origin]
    row=dict(task_id=t,record_id=i,origin=origin,sha256=h,trajectory_candidates=matches)
    if len(matches)==1:row.update(trajectory_id=matches[0][0],window_index=matches[0][1],trajectory_length=matches[0][2]);pool.append(row)
   frozen=selected_references(dict(task=task,records=records));cellrefs=[]
   for old in sorted(frozen,key=int):
    for r in frozen[old]:
     i=plan['immutable_feature_origins'].index(r['origin']);m=lineage[r['origin']]
     row=dict(r,task_id=int(old),record_id=i,constraint_index=len(cellrefs),trajectory_candidates=m,trajectory_id=m[0][0] if len(m)==1 else 'AMBIGUOUS',window_index=m[0][1] if len(m)==1 else None)
     cellrefs.append(row)
     if len(m)>1:ambiguity.append(row)
   if seed==46:
    actual=read(ROOT/f'reports/sample_qp_lifelong_v1_continuation_v4/plans/references_task{task}.json')
    assert [[r['origin'] for r in frozen[t]] for t in frozen]==[[r['origin'] for r in actual[t]] for t in frozen]
   selection[f'{seed}/{task}']=cellrefs;reforigins={r['origin'] for r in cellrefs}
   refdemos={t:{v[0] for r in cellrefs if r['task_id']==t for v in r['trajectory_candidates']} for t in range(6,task)}
   for s in group:
    sid=s['start_id'];batchorigins=[plan['immutable_feature_origins'][i] for i in s['record_indices']];batchdemos={t:{v[0] for o in batchorigins if int(Path(o).parent.name)==t for v in lineage[o]} for t in range(6,task+1)}
    old=[dict(r) for r in prior if r['start_id']==sid];used={r['origin'] for r in old}
    for j,r in enumerate(old):
     if r['origin'] not in reforigins:continue
     eligible=[r2 for r2 in pool if r2['task_id']==r['task_id'] and r2['origin'] not in used|reforigins and r2['trajectory_id'] not in batchdemos[r['task_id']]]
     eligible.sort(key=lambda q:(q['trajectory_id'] in refdemos[q['task_id']],q['trajectory_id']!=r['trajectory_id'],hashlib.sha256((sid+'/'+q['origin']).encode()).hexdigest()))
     assert eligible;replacement=dict(eligible[0],start_id=sid,replacement_for_origin=r['origin']);used.add(replacement['origin']);old[j]=replacement;changes.append(dict(start_id=sid,old_origin=r['origin'],new_origin=replacement['origin'],reason='exact reference-window overlap; identity-only deterministic replacement before outcomes'))
    for r in old:
     r['reference_trajectory_overlap']=r['trajectory_id'] in refdemos[r['task_id']];r['replacement_for_origin']=r.get('replacement_for_origin');r['prior_tau_reused']=r['replacement_for_origin'] is None
     old_all.append(r)
    available=[r for r in pool if r['task_id']==task and r['trajectory_id'] not in batchdemos[task]];groups=collections.defaultdict(list)
    for r in available:groups[r['trajectory_id']].append(r)
    local=random.Random(int(hashlib.sha256((sid+'/new_probe').encode()).hexdigest()[:16],16));demos=sorted(groups);local.shuffle(demos);assert demos
    for d in demos:local.shuffle(groups[d])
    for j in range(24):
     d=demos[j%len(demos)];assert groups[d];new_all.append(dict(groups[d].pop(),start_id=sid))
    refs_all.extend(dict(r,start_id=sid) for r in cellrefs)
 print('MEMBERSHIP',len(starts),len(old_all),len(new_all),'replacements',len(changes),'ambiguous refs',len(ambiguity),flush=True)
 write(OUT/'START_MANIFEST.json',dict(source_actual_update_fidelity_manifest=str(SOURCE/'START_MANIFEST.json'),source_sha256=sha(SOURCE/'START_MANIFEST.json'),**source))
 write(OUT/'REFERENCE_MANIFEST.json',dict(cells=selection,windows=refs_all,source_function='sample_qp_lifelong_v1_continuation_v4.common.selected_references',unchanged_hash_salt='sample_qp_lifelong_v1/46/{task}/{old_task}; reused literally on each seed retained buffer',seed46_matches_formal_manifest=True,ambiguous_lineage=ambiguity))
 write(OUT/'OLD_PROBE_MANIFEST.json',dict(windows=old_all,replacements=changes,source_sha256=sha(SOURCE/'PROBE_MANIFEST.json'),unchanged_count=len(old_all)-len(changes),reference_trajectory_overlap_count=sum(r['reference_trajectory_overlap'] for r in old_all),scope='window-disjoint; report actual trajectory overlap; no claim history-unseen',replacement_deadband='same frozen32eps rule, recomputed baseline only for122 new windows'))
 write(OUT/'NEW_PROBE_MANIFEST.json',dict(windows=new_all,trajectory_disjoint_from_current_batch=True))
 parity=[]
 for seed in (45,46,47):
  for task in (7,8):
   possible=[s for s in starts if s['seed']==seed and s['task_id']==task and s['anchor']['point']==s['start_id']];parity.append(random.Random(seed*100+task+20260923).choice(possible)['start_id'])
 files={str(p.relative_to(ROOT)):sha(p) for folder in ['sample_qp_optimized_v1_continuation_v4','sample_qp_lifelong_v1_continuation_v4','actual_update_fidelity_v1'] for p in (ROOT/folder).glob('*.py')}
 context=dict(branch=subprocess.check_output(['git','branch','--show-current']).decode().strip(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),dirty_tracked=subprocess.check_output(['git','status','--short','-uno']).decode(),python=sys.version,pytorch=torch.__version__,cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader']).decode().strip(),solver='v4 SLSQP->10k->20k->L-BFGS-B->trust-constr->fail-stop',kkt=solver.KKT_TOL,physical_tolerance=5e-6,trainable='same full decoder+head as120native MLR states',formal_training_resumed=False,formal_status=read(ROOT/'reports/sample_qp_lifelong_v1_continuation_v4/STATUS.json'),source_files=files,disk_free={p:shutil.disk_usage(p).free for p in ['/root','/data']})
 write(OUT/'EXPERIMENT_CONTEXT.json',context)
 protocol=dict(name='QP-Corrected Update Fidelity & Functional Workset Validation v1',phase1_starts=120,parity_starts=parity,kkt=1e-8,physical_tolerance=5e-6,reference_count='32 per old task;32 task7,64 task8',solver='unchanged v4; no fallback changes',d0='reuse exact prior candidate; formal core receives its original float32 displacement difference; final measured d_full is exact difference of actual float32 written parameters',solver_failure='preserve start, complete dump, no B result fabricated; any numerical failure prevents total gate PASS',deadband='reuse prior per-window tau exactly for2758 unchanged; same32eps baseline-noise formula for122 replacements and24new probes',primary_damage='mean positive actual old delta per start; median paired B-A',bootstrap=dict(replicates=10000,seed=2026092401,unit='start',stratified='resample20starts within each of6cells; pooled equal-cell median paired primary difference'),protection=dict(cells_negative_required=5,strong_reversal='cell median paired damage increase>10% cell median native positive damage plus mean old deadband',harmful_fraction='median paired fraction<0',introduced='introduced window count <=0.5*prevented count across all paired starts; report both rates and counts',CI='upper percentile95CI<0'),fidelity=dict(median_spearman=.9,median_sign=.9,median_recall=.8,median_precision=.8,stable_cells=5,defined_fraction=.5,decisive_coverage=.5),plasticity=dict(retention='median across native discernibly improving starts; each start mean probe delta',low='retention>=.9 and reversal<.1',severe='retention<.5 or reversal>.3',otherwise='MATERIAL_COST',no_eligible='unclassifiable; not fabricate LOW_COST'),phase2=dict(gate='protection AND corrected-fidelity SUPPORTED; all120solver/writebacks successful',K='floor(m/2)',methods=['FULL','RANDOM0','RANDOM1','RANDOM2','STRIDE','LATENT','GN'],rank=32,geometry='same mean-action GN head-only metric and q definition as H2; recompute at each state on full reference pool; frozen historical ACTION_METRIC diagonal; no share/conflict labels',latent='normalized mean over temporal and modality axes of H',kcenter='first canonical index then farthest min cosine distance; ties lowest canonical index',stride='sort(task,trajectory,window,origin), select floor((j+.5)*m/K)',constraint_generation='add single most positive normalized violated omitted constraint per round; m rounds maximum; full existing KKT and physical scan; no extra solver',timing='all feature/Jacobian+selection+repeated QP+full scans; report shared H construction separately',random_replicates=[0,1,2],gn_gate='>=5of6cells better than BOTH stride and latent on at least2 metrics with no material extra plasticity; total GN time counted; exact efficacy ties do not count',comparison_eps=dict(update_rel=1e-5,loss='frozen deadband'),resource='no full120model-state copies; phase2 reconstructs same frozen origins again if required'),formal_training_resumed=False)
 write(OUT/'PROTOCOL.json',protocol)
if __name__=='__main__':main()
