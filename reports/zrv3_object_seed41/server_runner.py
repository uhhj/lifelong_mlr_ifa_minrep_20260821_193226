"""Finite v3 -> conditional v3-A sequence. Never restarts a partial phase."""
import json, os, sys, time, hashlib, shutil, subprocess, traceback, shlex
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from flow_metrics import metrics, pack
N=Path(__file__).resolve().parent
PY='/data/mlr_ifa/env/bin/python'
OLD=Path('/data/mlr_ifa/experiment_workspaces/RA_ZRv2_autonomous_20261019/zr_runs/OBJECT_RA_seed41')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(x,indent=2)+'\n');os.replace(q,p)
def paths(v):return N/v,Path('/data/mlr_ifa/experiments/ZRv3_OBJECT_RA_seed41_20261020_'+v)
def verify():
    for n,h in js(N/'ORCHESTRATION.json')['files'].items():assert sha(N/n)==h,('orchestration changed',n)
def workers():
    hits=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            c=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            env=(p/'environ').read_bytes()
            if str(N).encode() in env and any(x in c for x in ('-m zr3_exp.calibrate','-m zr3_exp.evaluate','-m zr3_exp.audit')):hits.append(c)
        except (OSError,UnicodeError):pass
    return hits
def prepare(v):
    verify();assert v in ('v3','v3a');W,O=paths(v)
    assert not W.exists() and not O.exists(),'No duplicate preparation'
    assert shutil.disk_usage('/data').free>2*1024**3
    spec=js(OLD/'RUN_SPEC.json')
    assert js(spec['parent_engineering_file'])['status']=='PASS'
    shutil.copytree(N/'template',W);O.mkdir()
    spec.update(variant=v,arm='ra_coreset_ifa_zr'+v,output=str(O),epsilon=.5 if v=='v3' else None,
                controller_contract=str(N/'ORCHESTRATION.json'),created=time.time(),
                gate='AUC>parent; FWT>=parent; OldFinal>=parent; engineering PASS',
                new_uv_initialization=True,old_zrv2_output=js(OLD/'RUN_SPEC.json')['output'])
    write(W/'RUN_SPEC.json',spec)
    (W/'env.sh').write_text('#!/usr/bin/env bash\nsource '+spec['parent_env']+'\nexport PYTHONPATH="'+str(W/'source')+':'+str(W/'attachment')+':$PYTHONPATH"\nexport PYTHONDONTWRITEBYTECODE=1\ncd '+str(W)+'\n')
    write(O/'RUN_STATUS.json',dict(status='PREPARING',time=time.time()))
    return W,O
def phase(v,name,args,envvars=None):
    verify();W,O=paths(v);L=O/'preflight' if name not in ('pipeline','finish') else O
    L.mkdir(exist_ok=True)
    assert not (L/(name+'.exit')).exists() and not (L/(name+'.log')).exists(),('Already attempted',v,name)
    command=['bash','-c','source '+shlex.quote(str(W/'env.sh'))+' && '+(' '.join(k+'='+shlex.quote(val) for k,val in (envvars or {}).items())+' ' if envvars else '')+shlex.join(args)]
    with (L/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),command=command,phase=name))+'\n')
    with (L/(name+'.log')).open('x') as log:
        r=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (L/(name+'.exit')).write_text(str(r.returncode)+'\n')
    assert r.returncode==0,(v,name,r.returncode)
def preflight(v):
    W,O=paths(v)
    for name,args,env in [
        ('software_reference',[PY,str(W/'attachment/test_reference.py')],None),
        ('software_exec_cpu',[PY,str(W/'attachment/test_execution_core.py')],{'TEST_DEVICE':'cpu'}),
        ('software_exec_cuda',[PY,str(W/'attachment/test_execution_core.py')],{'TEST_DEVICE':'cuda'}),
        ('admission',[PY,str(W/'admission.py')],None),
        ('integration',[PY,'-m','zr3_exp.integration'],None),
        ('cost',[PY,'-m','zr3_exp.cost_and_isolation'],None),
        ('freeze',[PY,'-m','zr3_exp.freeze'],None)]:
        phase(v,name,args,env)
    write(O/'PREFLIGHT_ACCEPTANCE.json',dict(status='PASS',optimizer_calls=4,environment_steps=0,time=time.time()))
def finish(v):
    verify();W,O=paths(v);s=js(W/'RUN_SPEC.json');P=Path(s['parent'])
    assert (O/'pipeline.exit').read_text().strip()=='0' and not workers()
    assert all((O/(x+'.exit')).read_text().strip()=='0' for x in ('calibrate','evaluate','audit'))
    a=js(O/'AUDIT.json');assert a['status']=='PASS' and a['epochs']==40 and a['optimizer_steps']==10550
    assert js(O/'CHECKPOINT_LOAD_ACCEPTANCE.json')['status']=='PASS'
    assert len(js(O/'CHECKPOINT_LOAD_ACCEPTANCE.json')['rows'])==4
    for f,h in js(O/'CONTRACT.json')['files'].items():assert sha(f)==h,f
    pm,pr,prows,ph=metrics(P/'ra_coreset/evaluation',41,'ra_coreset')
    nm,nr,nrows,nh=metrics(O/'evaluation',41,s['arm'])
    assert ph==s['baseline_episode_hashes']
    assert (pm['AUC'],pm['FWT'],pm['OldFinal'])==(F(701,960),F(3,4),F(47,60))
    for k in prows:
        assert prows[k]['checkpoint_sha256']==nrows[k]['parent_checkpoint_sha256']
        for f in ('initial_state_index','initial_state_sha256','modal_sha256'):assert prows[k][f]==nrows[k][f]
    events=Counter(json.loads(l)['event'] for p in (O/'evaluation').glob('attempts*.jsonl') for l in p.read_text().splitlines())
    assert events==Counter(vector_attempt_start=20,vector_attempt_complete=20,environment_construct_start=10,environment_construct_complete=10,environment_closed=10)
    passed=nm['AUC']>pm['AUC'] and nm['FWT']>=pm['FWT'] and nm['OldFinal']>=pm['OldFinal']
    assert passed==js(O/'METRIC_COMPARISON.json')['metric_gate']
    # Historical v2 results are comparison-only, never used for restart or sampling.
    vm,vr,vrows,vh=metrics(Path(s['old_zrv2_output'])/'evaluation',41,'ra_coreset_ifa_zrv2')
    result=dict(status='PASS',variant=v,gate_passed=passed,parent=pack(pm),new=pack(nm),old_v2=pack(vm),
                delta_parent=pack({k:nm[k]-pm[k] for k in nm}),delta_v2=pack({k:nm[k]-vm[k] for k in nm}),
                matrices={name:[[float(r[st,t]*100) if t<=st else None for t in range(6,10)] for st in range(6,10)] for name,r in [('parent',pr),('new',nr),('v2',vr)]},
                epochs=40,optimizer_steps=10550,checkpoints=4,records=200,vector_attempts=20,
                no_scientific_workers=True,input_hashes_pass=True,episode_hashes=nh,old_v2_episode_hashes=vh,time=time.time())
    write(O/'PROCESS_EXIT_ACCEPTANCE.json',result);write(O/'RUN_STATUS.json',dict(status='COMPLETED',gate_passed=passed,time=time.time()))
    return result
def main():
    mode=sys.argv[1]
    if mode=='prepare':
        prepare(sys.argv[2]);return
    if mode=='preflight':
        preflight(sys.argv[2]);return
    if mode=='finish':
        print(json.dumps(finish(sys.argv[2])));return
    assert mode=='run' and not (N/'controller.started').exists() and not workers()
    verify();write(N/'controller.started',dict(pid=os.getpid(),time=time.time()))
    (N/'controller.pid').write_text(str(os.getpid()));code=1
    try:
        results={}
        for v in ('v3','v3a'):
            W,O=paths(v)
            if v=='v3a':prepare(v);preflight(v)
            assert js(O/'PREFLIGHT_ACCEPTANCE.json')['status']=='PASS'
            # pipeline.py owns pipeline.exit/log. The launcher has distinct names.
            phase(v,'formal',[PY,str(W/'pipeline.py')])
            results[v]=finish(v)
            write(N/'RESULTS.json',results)
            if v=='v3' and not results[v]['gate_passed']:
                write(N/'BRANCH_DECISION.json',dict(v3_gate=False,next='STOP',reason='Frozen single-seed screen failed',time=time.time()));break
            if v=='v3':write(N/'BRANCH_DECISION.json',dict(v3_gate=True,next='v3a',reason='Required objective-only ablation',time=time.time()))
        write(N/'FINAL_ACCEPTANCE.json',dict(status='PASS',variants=list(results),results=results,no_workers=not workers(),time=time.time()))
        assert not workers()
        write(N/'RUN_STATUS.json',dict(status='COMPLETED',variants=list(results),time=time.time()));code=0
    except BaseException:
        write(N/'BLOCKED.json',dict(status='BLOCKED',time=time.time(),traceback=traceback.format_exc()))
        write(N/'RUN_STATUS.json',dict(status='BLOCKED',time=time.time()))
        raise
    finally:(N/'controller.exit').write_text(str(code)+'\n')
if __name__=='__main__':main()
