"""Adapt immutable accepted sources into a distinct finite fork root."""
from pathlib import Path
import json,ast,hashlib,sys,difflib
R=Path(__file__).resolve().parent;NAME='SUPPORT_GATE_FORKS_20261027_v1';ROOT='/data/mlr_ifa/experiment_workspaces/'+NAME
pre=json.loads((Path(sys.argv[1])/'EVIDENCE.json').read_text(encoding='utf8'));assert pre['status']=='PASS_READONLY_ASSETS'
bundle={};review={}
def put(n,s):
    p=R/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(s,encoding='utf8');bundle[n]=s
    if n.endswith('.py'):ast.parse(s,feature_version=(3,8))
for name,c in pre['cases'].items():
    parent=R/'reference_source'/name
    for n in ['runtime.py','layout.py','losses.py','legacy_losses.py','evaluate.py','updates.py','memory.py']:
        old=(parent/n).read_text(encoding='utf8');s=old
        if n=='layout.py':
            s=s[:s.index('ARMS=')]+f"ARMS={c['arms']!r}\nSEED={c['seed']}\nSTAGE={c['stage']}\nCASE={name!r}\nNAME={NAME!r}\n"+"ROOT_PRIVATE=Path('/root/mlr_ifa/experiments')/NAME\nDATA_PRIVATE=Path('/data/mlr_ifa/experiments')/NAME\nOUTPUT={a:DATA_PRIVATE/CASE/a for a in ARMS}\nVOLUME_BOUND={'/root':0,'/data':8*1024**3}\n"+'\n'.join(k+'='+repr({int(t):v for t,v in c[k.lower()].items()}) for k in ['POP','VALID','OLD'])+"\nMEMORY_PER_TASK=128\nANCHOR_BATCH=4\nRETAIN_WEIGHT=.1\ndef model_path(arm,stage):return OUTPUT[arm]/('stage%d.pth'%stage)\ndef memory_path(arm,task):return Path(BINDING['caches'][str(task)]['path'])\n"
        elif n=='runtime.py':
            pos=s.index('\ndef capacity():');s=s[:pos]+s[pos:]
            start=s.index('def verify():');end=s.index('def capacity():')
            s=s[:start]+"BINDING=js(R/'BINDING.json')\ndef verify():\n contract=js(R/'CONTRACT.json')\n for path,h in contract['files'].items():assert sha(path)==h,path\n for path,h in BINDING['files'].items():assert sha(path)==h,path\n assert all(sha(S/n)==h for n,h in BINDING['parent_source_sha256'].items())\n return sha(R/'CONTRACT.json')\n\n"+s[end:]
        elif n=='updates.py':
            put(name+'/legacy_updates.py',s.replace('from memory import retention','from legacy_memory import retention'))
            s=s.replace('from memory import retention','from support_memory import retention').replace('retention(a.policy.module,memory,stage,a.step)','retention(a.policy.module,memory,stage,a.step,arm)')
        elif n=='memory.py':
            put(name+'/legacy_memory.py',s);continue
        elif n=='evaluate.py':
            budget=len(c['arms'])*(c['stage']-5)*20
            s=s.replace("load('S',6,c,shape)","load(ARMS[0],STAGE,c,shape)").replace("load('D',6,c,shape)","load(ARMS[0],STAGE,c,shape)")
            s=s.replace('control_seed(41)','control_seed(SEED)').replace('41*1000','SEED*1000').replace('41+group','SEED+group')
            s=s.replace('range(6,10)','[STAGE]')
            import re
            s=re.sub(r'budget=\d+',f'budget={budget}',s);s=re.sub(r'len\(records\)==\d+',f'len(records)=={budget}',s)
            s=re.sub(r'episodes=\d+,vectors=\d+,cells=\d+',f'episodes={budget},vectors={budget//10},cells={budget//20}',s)
        put(name+'/'+n,s)
        review[name+'/'+n]=dict(parent_sha256=hashlib.sha256(old.encode()).hexdigest(),sha256=hashlib.sha256(s.encode()).hexdigest(),unchanged=old==s)
        if old!=s:(R/'review').mkdir(exist_ok=True);(R/'review'/(name+'_'+n+'.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True))),encoding='utf8')
    for source,target in [('train_fork.py','train.py'),('fork_tools.py','fork_tools.py'),('support_memory.py','support_memory.py')]:put(name+'/'+target,(R/source).read_text(encoding='utf8'))
    put(name+'/BINDING.json',json.dumps(c,indent=2)+'\n')
for n in ['controller.py','PROTOCOL.md','CASES.json']:put(n,(R/n).read_text(encoding='utf8'))
put('PREDEPLOY.json',json.dumps(dict(observed=pre['observed'],capacity=pre['capacity'],hardware=pre['hardware'],review=review),indent=2)+'\n')
(R/'DEPLOY_TEXT_BUNDLE.json').write_text(json.dumps(bundle,ensure_ascii=True,indent=2)+'\n',encoding='utf8')
(R/'UPLOAD_REVIEW.json').write_text(json.dumps(dict(status='TEXT_ONLY_REVIEWED_PENDING_RUNTIME',files={n:dict(bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()) for n,s in bundle.items()},source_review=review),indent=2)+'\n',encoding='utf8')
print('Prepared',len(bundle),'files; no remote dispatch')
