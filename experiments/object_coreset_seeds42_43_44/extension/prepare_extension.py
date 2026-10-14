"""Materialize isolated seed43/44 runners; never edit completed seed42."""
import pathlib,json,hashlib,shutil,difflib,tarfile
P=pathlib.Path(__file__).resolve().parent
BASE=P.parent/'20261013_coreset_object'
contract=json.loads((BASE/'launch_evidence/CONTRACT.json').read_text())
for name,h in contract['source_sha256'].items():
    assert hashlib.sha256((BASE/'source'/name).read_bytes()).hexdigest()==h,name
for seed in (43,44):
    dest=P/f'seed{seed}'/'source'
    assert not dest.exists()
    shutil.copytree(BASE/'source',dest,ignore=shutil.ignore_patterns('__pycache__'))
    def change(name,old,new):
        p=dest/'coreset_exp'/name;s=p.read_text(encoding='utf-8');assert old in s,(name,old)
        p.write_text(s.replace(old,new),encoding='utf-8',newline='\n')
    change('common.py',"O=Path(os.environ['CORESET_OUT'])", "O=Path(os.environ['CORESET_OUT'])\nSEED=int(os.environ['CORESET_SEED'])\nassert SEED in (43,44), 'Only explicitly authorized extension seeds'\nSELECTOR_SEED=42 # Frozen calibration/reference subset setting, not training RNG")
    for name in ('configure.py','export.py','train.py','preflight.py','storage.py','evaluate.py','summarize.py','acceptance.py'):
        p=dest/'coreset_exp'/name;s=p.read_text(encoding='utf-8');p.write_text('from .common import SEED,SELECTOR_SEED\n'+s,encoding='utf-8',newline='\n')
    change('configure.py',"'seed=42'","f'seed={SEED}'")
    change('configure.py',"protocol='paper_aligned_seed42_only_v2',seed=42", "protocol='paper_aligned_seed_extension_v2',seed=SEED")
    change('configure.py',"extensions=[],warmup_provenance=", "extensions=[43,44],selector_seed=SELECTOR_SEED,extension_authorization='User requested remaining two seeds after seed42 completion; no other scientific changes',warmup_provenance=")
    for name in ('export.py','train.py','preflight.py','evaluate.py'):
        change(name,'control_seed(42)','control_seed(SEED)')
    change('train.py','c.buffer_dir,42,start=6','c.buffer_dir,SEED,start=6')
    change('preflight.py','env.seed(42)','env.seed(SEED)')
    change('storage.py','random.Random(42)','random.Random(SEED)')
    change('storage.py','greedy_kcenter_select(paths,k,42,population)','greedy_kcenter_select(paths,k,SELECTOR_SEED,population)')
    change('evaluate.py','see=42','see=SEED')
    change('summarize.py',"row['seed']==42","row['seed']==SEED")
    change('summarize.py','==[42,43]','==[SEED,SEED+1]')
    for name in ('summarize.py','acceptance.py'):change(name,'seed=42','seed=SEED')
    change('summarize.py',"'# Seed42 Random vs Coreset'","f'# Seed{SEED} Random vs Coreset'")
    change('summarize.py','不扩展seed，不更新旧实验指标，不执行后续调参。','仅执行用户授权的43/44扩展；不更新旧实验指标，不执行后续调参。')
    change('env.sh','Coreset_paper_OBJECT_seed42_20261013',f'Coreset_paper_OBJECT_seed{seed}_20261013')
    change('env.sh','export PYTHONPATH=',f'export CORESET_SEED={seed}\nexport PYTHONPATH=')
    for name in ('pipeline.sh','supervise.sh'):
        change(name,'Coreset_paper_OBJECT_seed42_20261013',f'Coreset_paper_OBJECT_seed{seed}_20261013')
    change('pipeline.sh','38315',str(38315+seed-42))
    # Validate the seed contract in every entry point, including restored paths.
    change('model.py','def cfg(name):return EasyDict(json.loads((O/f\'resolved_{name}.json\').read_text()))',
        "def cfg(name):\n    from .common import SEED\n    c=EasyDict(json.loads((O/f'resolved_{name}.json').read_text()))\n    assert c.seed==SEED and all(c.policy[k] is True for k in ('use_vision','use_language','use_state'))\n    return c")
    patch=[];changes=[]
    for f in sorted(dest.rglob('*')):
        if not f.is_file():continue
        rel=f.relative_to(dest);old=BASE/'source'/rel
        if old.read_bytes()!=f.read_bytes():
            changes.append(str(rel));patch.extend(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),f.read_text(encoding='utf-8').splitlines(True),fromfile='seed42/'+str(rel),tofile=f'seed{seed}/'+str(rel)))
    (P/f'seed{seed}'/'source_vs_seed42.patch').write_text(''.join(patch),encoding='utf-8')
    (P/f'seed{seed}'/'changes.json').write_text(json.dumps(changes,indent=2))
    with tarfile.open(P/f'seed{seed}'/'source.tar.gz','w:gz') as t:t.add(dest,arcname='source')
    monitor=(BASE/'monitor_snapshot.py').read_text().replace('Coreset_paper_OBJECT_seed42_20261013',f'Coreset_paper_OBJECT_seed{seed}_20261013')
    (P/f'seed{seed}'/'monitor_snapshot.py').write_text(monitor,encoding='utf-8',newline='\n')
print('Isolated sources built from verified seed42 frozen files; model/loss/selector unchanged.')
