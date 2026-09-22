"""Sequential, resumable, fixed-protocol CPU timing experiment."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse,csv,json,subprocess,sys,time,traceback,fcntl
from pathlib import Path
import numpy as np
import scipy
from qp_workingset_refinement_diag_v1.numeric import write,save,sha,array_sha
from .data import fixture_specs,synthetic,real,SEEDS,ROOT
from .grouping import group
from .engine import run as execute
from .reference import make as reference,compare
OUT=ROOT/'reports/qp_similarity_workingset_offline_v1'
TAUS=(.999,.99,.9999)
ARMS=[('FULL_WS',None),('SIM_ADD',None),('RANDOM_ADD',0),('RANDOM_ADD',1),('RANDOM_ADD',2)]

def read(p):return json.loads(Path(p).read_text())
def name(arm,seed):return arm if seed is None else arm+'_'+str(seed)
def plain_first(first):return {k:v for k,v in first.items() if k!='x'}

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    files={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    for n in ('refine.py','numeric.py','run.py'):
        p=ROOT/'qp_workingset_refinement_diag_v1'/n;files[str(p.relative_to(ROOT))]=sha(p)
    protocol=dict(version=1,primary_tau=.999,sensitivity=[.99,.9999],seeds=list(SEEDS),
        implementation_seeds=list(SEEDS[:2]),confirmation_seed=SEEDS[2],
        m=[32,64,96],p=128,synthetic_count=27,real_count=5,
        synthetic_rng='Fresh private numpy RandomState(seed) per kind/m case; no global RNG',
        high='m/4 Gaussian-QR centers, positive diag(R) signs; 4 unit-noise(.01) samples per center',
        mixed='m/8 Gaussian-QR centers; 4 positive and 2 near-negative samples per center; m/4 independent columns',
        v='unit Gaussian drawn after all columns from same local stream',
        grouping='signed complete-link admissibility in valid order; max mean group affinity, tie group ID; true medoid tie valid ID',
        main_initial_x='zero for every branch; references never passed to algorithm',
        random_seeds=[0,1,2],random_k='SIM cluster count design input; no grouping time charged to RANDOM',
        warmups=1,repetitions=10,order='cyclic rotation by fixture index + threshold index + repeat index, sequential',
        blas_threads=1,cpu_affinity='single first currently allowed logical CPU',
        total_timing='external wall clock around entire engine call: selection, construction, kernel, full certification',
        excluded_timing=['file IO','reference preparation','extra spectral diagnostics','E_update comparison','reporting'],
        inherited_kernel='unchanged qp_workingset_refinement_diag_v1.refine.run',
        kernel_limits=dict(working_set_iterations=512,linear_refinements=3),outer_limit='m subQP calls',
        reinsertion='all inactive coordinates with float64 OR Decimal80 residual >1e-8; never shrink outer C',
        full_check_reuse='reuse inner certification only when C is the full original index order, for ALL arms',
        first_round='SIM_ONESHOT is the SIM_ADD first result/time; never separately solve',
        kkt=1e-8,reference_kkt=1e-12,E_update=1e-5,
        engineering_budgets_not_universal_safety=True,
        reference='cached five full solutions or full zero-start kernel; one stationary full-support refinement, <=1e-12 else unresolved',
        reference_never_initializes_competitors=True,
        negative_quadratic='report error if q80 < -1e-60*max(1,sum(abs(terms))); log smaller negatives explicitly',
        N_crosscheck_budget='abs difference <=1e-10 +1e-7*max(two norms); arithmetic audit only',
        zero_u='physical zero correction; normalized E_update not defined; fixed main fixtures require u_norm>0',
        rules=dict(H1='SIM_ADD all32 strict KKT + E_update pass, boundary tests expected outcomes',
            H2='high/mixed median k_final/m <=.75',
            H3='high/mixed median SIM/random-median T <=.90 OR solve-count ratio <=.80 AND T ratio <=1',
            H4='high/mixed median SIM/FULL T <=.80 AND low <=1.10',
            real_advance='all5 strict accuracy AND geometric mean SIM/FULL T <=.90',
            heavy_label='high/mixed median k_final/m >.90; descriptive secondary marker'),
        optional_real_32_96='not added; no asset search beyond fixed fixture paths',
        flags=dict(training_active=False,formal_solver_modified=False,model_writeback=False,GPU_required=False),
        sources=files,numpy=np.__version__,scipy=scipy.__version__,
        source_commit='4c55fb42e3cb42f66ce1a626614d492ed77a1984')
    dest=OUT/'PROTOCOL.json'
    if dest.exists():assert read(dest)==protocol,'Frozen code/protocol mismatch'
    else:write(dest,protocol)
    specs=fixture_specs();manifest=[]
    for spec in specs:
        entry=dict(spec)
        if spec['source']=='real':
            p=Path(spec['path']);entry['sha256']=sha(p)
            entry['captured_stage']='post10k before continuation' if spec['id']=='F1' else 'post30k standard matrix (container may include later candidates)'
            ref=ROOT/'reports/qp_workingset_refinement_diag_v1/candidates'/('regression_'+spec['id'])/'candidate.npz'
            entry['reference_path']=str(ref);entry['reference_sha256']=sha(ref)
        manifest.append(entry)
    if (OUT/'FIXTURES.json').exists():assert read(OUT/'FIXTURES.json')==manifest
    else:write(OUT/'FIXTURES.json',manifest)
    return protocol,manifest

def validate_sources(protocol):
    for p,h in protocol['sources'].items():assert sha(ROOT/p)==h,('Frozen source changed',p)

def dataset(spec):
    p=OUT/'inputs'/(spec['id']+'.npz')
    if not p.exists():
        z=real(spec['path']) if spec['source']=='real' else synthetic(spec['m'],spec['kind'],spec['seed'])
        save(p,**z)
    with np.load(p,allow_pickle=False) as f:z={k:f[k].copy() for k in f.files}
    assert float(z['u_norm'])>0,'Zero-native case: normalized experiment inapplicable; physical correction is zero'
    return z

def get_reference(spec,z):
    d=OUT/'references'/spec['id'];d.mkdir(parents=True,exist_ok=True)
    if (d/'RESULT.json').exists():return np.load(d/'reference.npz')['x'],read(d/'RESULT.json')
    cached=None
    if spec['source']=='real':
        with np.load(spec['reference_path']) as q:
            assert np.array_equal(q['A'],z['A']) and np.array_equal(q['b'],z['b']);cached=q['x'].copy()
    x,meta=reference(z['A'],z['b'],cached)
    save(d/'reference.npz',x=x,A=z['A'],b=z['b']);write(d/'RESULT.json',meta)
    return x,meta

def measure(spec,z,ref,refmeta,tau,arm,seed,k,repeat,cell):
    base=cell/name(arm,seed);base.mkdir(parents=True,exist_ok=True)
    target=base/('warmup.json' if repeat<0 else 'repeat%02d.json'%repeat)
    if target.exists():return
    t=time.perf_counter();result=execute(z['A'],z['b'],z['valid'],arm,tau,seed,k)
    total=time.perf_counter()-t;meta=result['meta'];meta['total_seconds']=total
    row=dict(meta,fixture=spec['id'],arm=arm,random_seed=seed,tau=tau,repeat=repeat,
        x_sha256=array_sha(result['x']),first=plain_first(result['first']))
    if repeat==0:
        validation=compare(z['A'],z['b'],result['x'],ref,z.get('N')) if refmeta['status']=='PASS' else dict(status='REFERENCE_UNRESOLVED',update_budget_pass=False)
        row['validation']=validation
        if arm=='SIM_ADD':
            row['oneshot_validation']=compare(z['A'],z['b'],result['first']['x'],ref,z.get('N')) if refmeta['status']=='PASS' else dict(status='REFERENCE_UNRESOLVED',update_budget_pass=False)
        save(base/'candidate.npz',x=result['x'],first_x=result['first']['x'])
        write(base/'constraint_trace.json',result['trace']);write(base/'linear_systems.json',result['linear_systems'])
    elif repeat>0:
        initial=read(base/'repeat00.json')
        row['deterministic_match']=all(row[key]==initial[key] for key in ('x_sha256','status','k0','k_final','linear_solves','subqp_rounds','float64','decimal80'))
    write(target,row)

def main():
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[0]})
    protocol,specs=freeze()
    unit=read(OUT/'UNIT_TESTS.json')
    assert unit['passed'] and unit['tests_sha256']==sha(Path(__file__).with_name('tests.py'))
    lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(OUT/'RUNTIME.json',dict(pid=os.getpid(),cpu_affinity=list(os.sched_getaffinity(0)),
        cpu=os.uname().machine,load_at_start=list(os.getloadavg()),python=sys.version,
        implementation_sha=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        **protocol['flags']))
    completed=0
    # Primary all32 first; sensitivity only after main confirmation, no tuning between.
    for ti,tau in enumerate(TAUS):
        for fi,spec in enumerate(specs):
            validate_sources(protocol);cell=OUT/'cases'/spec['id']/('tau_'+str(tau))
            if (cell/'COMPLETE.json').exists():completed+=1;continue
            z=dataset(spec);ref,rm=get_reference(spec,z)
            k=len(group(z['A'],z['valid'],tau)[0])
            cell.mkdir(parents=True,exist_ok=True)
            write(OUT/'STATUS.json',dict(status='RUNNING',fixture=spec['id'],split=spec['split'],tau=tau,
                cells_complete=completed,cells_total=len(specs)*len(TAUS),pid=os.getpid(),time=time.time(),**protocol['flags']))
            for rep in range(-1,10):
                shift=(fi+ti+max(rep,0))%len(ARMS);order=ARMS[shift:]+ARMS[:shift]
                for arm,seed in order:measure(spec,z,ref,rm,tau,arm,seed,k,rep,cell)
            write(cell/'COMPLETE.json',dict(status='MEASURED',fixture=spec,tau=tau,k_design=k,
                reference_status=rm['status'],input_sha256=sha(OUT/'inputs'/(spec['id']+'.npz')),
                A_symmetry_error=float(np.max(np.abs(z['A']-z['A'].T))),cond_A=float(np.linalg.cond(z['A'])),
                time=time.time()))
            completed+=1;print('CELL_COMPLETE',completed,'/',len(specs)*len(TAUS),spec['id'],tau,flush=True)
            from .summarize import summarize
            summarize(OUT,complete=False)
    validate_sources(protocol)
    audit=[]
    for spec in specs:
        if spec['source']=='real':
            for field,hfield in [('path','sha256'),('reference_path','reference_sha256')]:
                actual=sha(Path(spec[field]));audit.append(dict(path=spec[field],unchanged=actual==spec[hfield],sha256=actual))
    write(OUT/'READONLY_AUDIT.json',dict(inputs=audit,all_unchanged=all(x['unchanged'] for x in audit)))
    assert all(x['unchanged'] for x in audit)
    from .summarize import summarize
    summarize(OUT,complete=True)
    write(OUT/'STATUS.json',dict(status='COMPLETE',cells_complete=completed,time=time.time(),**protocol['flags']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');a=parser.parse_args()
    if a.output:OUT=Path(a.output).resolve()
    try:main()
    except BaseException:
        write(OUT/'ERROR.json',dict(error=traceback.format_exc(),time=time.time(),training_active=False,
            formal_solver_modified=False,model_writeback=False,GPU_required=False));raise
