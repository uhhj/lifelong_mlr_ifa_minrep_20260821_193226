"""Derive isolated GOAL runners from the hash-verified completed OBJECT runner."""
import json,hashlib,shutil,difflib,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent
P=R.parent/'20261013_coreset_extension'
B=P/'seed43/source'
C=json.loads((P/'launch_evidence/seed43/CONTRACT.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in C['source_sha256'].items():assert sha(B/n)==h,n
Q='/root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017'
PT_HASH='241d3370acfc04d694df04256956a97aa13cfdc2e651cf258d338f8e72988be7'
for seed in (42,43,44):
    D=R/f'seed{seed}/source';assert not D.exists();shutil.copytree(B,D,ignore=shutil.ignore_patterns('__pycache__'))
    def change(n,old,new):
        p=D/'coreset_exp'/n;s=p.read_text(encoding='utf-8');assert old in s,(n,old);p.write_text(s.replace(old,new),encoding='utf-8',newline='\n')
    change('common.py','assert SEED in (43,44)','assert SEED in (42,43,44)')
    for n in ('configure.py','train.py'):change(n,'LIBERO_OBJECT','LIBERO_GOAL')
    change('configure.py','object_multitask_model_ep99.pth','goal_multitask_model_ep96.pth')
    change('configure.py','159351b1df166dd5391c6b1e1317c0ab299fc8b1447fe466acef14ed7896bb7b',PT_HASH)
    change('configure.py','+gamma=0.3','+gamma=0.7');change('configure.py','gamma=.3','gamma=.7')
    change('configure.py',"'benchmark_name=LIBERO_GOAL',","'benchmark_name=LIBERO_GOAL','data.task_order_index=0','data.demo_limit=10','data.seq_len=8',")
    change('configure.py',"for root in ('/data/mlr_ifa/experiments','/data/mlr_ifa/analyses'):","for root in ('/data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/run', *[f'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{s}_20261013' for s in (42,43,44)]):")
    change('configure.py',"extensions=[43,44]","extensions=[42,43,44]")
    change('configure.py','paper_aligned_seed_extension_v2','paper_aligned_GOAL_three_seed_v1')
    change('configure.py','User requested remaining two seeds after seed42 completion; no other scientific changes','User authorized GOAL Random/Coreset seeds42/43/44; system disk outputs; fresh paired baselines due disclosed historical differences')
    change('configure.py',"created_unix=time.time()","created_unix=time.time(),suite='LIBERO_GOAL',expected_windows=[997,889,904,1969],expected_steps_per_arm=76100,orchestration_sha256=sha(S.parents[1]/'ORCHESTRATION_SHA256.json')")
    change('common.py',"assert source_hashes()==c['source_sha256'],'Frozen source mismatch'","assert source_hashes()==c['source_sha256'],'Frozen source mismatch'\n    orchestration=S.parents[1]/'ORCHESTRATION_SHA256.json'\n    assert sha(orchestration)==c['orchestration_sha256']\n    for p,h in json.loads(orchestration.read_text()).items():assert sha(p)==h,p")
    change('preflight.py','angular_mean(g,own,[old],.3)','angular_mean(g,own,[old],.7)')
    change('preflight.py','angular_mean(g,own,[old,old],.3)','angular_mean(g,own,[old,old],.7)')
    change('preflight.py','angular_mean(g,own,[],.3)','angular_mean(g,own,[],.7)')
    change('export.py',"write(O/'FEATURES.json'","assert counts=={'6':997,'7':889,'8':904,'9':1969},counts\n        write(O/'FEATURES.json'")
    change('summarize.py',"for arm in ('random','coreset'):","assert totalsteps==76100\n    for arm in ('random','coreset'):")
    change('summarize.py','仅执行用户授权的43/44扩展','仅执行用户授权的GOAL三seed42/43/44')
    change('summarize.py',"extensions=[]","extensions=[42,43,44]")
    oldS='/data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_seed43_20261013/source'
    newS=Q+f'/seed{seed}/source';newO=f'/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed{seed}_20261017'
    for n in ('env.sh','pipeline.sh','supervise.sh'):change(n,oldS,newS)
    change('env.sh','/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed43_20261013',newO)
    change('env.sh','export CORESET_SEED=43',f'export CORESET_SEED={seed}')
    change('pipeline.sh','38316',str(39342+seed-42))
    # Model loading must reject wrong suite, weight or gamma on every entry.
    change('model.py',"assert c.seed==SEED", "assert c.benchmark_name=='LIBERO_GOAL' and c.gamma==.7 and c.pretrain.endswith('goal_multitask_model_ep96.pth')\n    assert c.seed==SEED")
    patch=[]
    for p in sorted(D.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(D);old=B/rel
        if p.read_bytes()!=old.read_bytes():patch.extend(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),p.read_text(encoding='utf-8').splitlines(True),fromfile='OBJECT/'+str(rel),tofile='GOAL/'+str(rel)))
        if p.suffix=='.py':compile(p.read_text(encoding='utf-8'),str(p),'exec')
    (D.parent/'source_vs_OBJECT.patch').write_text(''.join(patch),encoding='utf-8',newline='\n')
    (D.parent/'SOURCE_SHA256.json').write_text(json.dumps({p.relative_to(D).as_posix():sha(p) for p in D.rglob('*') if p.is_file()},indent=2))
    # Model, selector, optimizer core and rollout function have no suite-specific edits.
    for n in ('libero/lifelong/models/complete_policy.py','libero/lifelong/models/brief.py','libero/lifelong/algos/base.py','libero/lifelong/algos/mlr.py','libero/lifelong/selectors/greedy_kcenter.py','coreset_exp/eval_metric.py','coreset_exp/storage.py'):
        assert sha(D/n)==sha(B/n),n
(R/'DERIVATION.json').write_text(json.dumps(dict(status='PASS',base_commit='a721807f2f7fa7f9833dcc85c592d78e9accd00b',source_parent=str(B),parent_contract_sha256=sha(P/'launch_evidence/seed43/CONTRACT.json'),seeds=[42,43,44],goal_weight_sha256=PT_HASH),indent=2))
print('Three isolated GOAL sources compiled; frozen model/loss core/selector/rollout bytes preserved')
