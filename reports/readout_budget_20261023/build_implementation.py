"""Recorded mechanical derivation of a new budget-only source; never modifies v2."""
from pathlib import Path
import ast, hashlib, json
R=Path(__file__).resolve().parent
V=R.parent/'20261023_readout_patch_v2'
Q='/data/mlr_ifa/experiment_workspaces/READOUT_BUDGET_OBJECT41_20261023_v1'
VQ='/data/mlr_ifa/experiment_workspaces/READOUT_PATCH_OBJECT41_20261023_v2'
(R/'pbd').mkdir(exist_ok=True)
def source(n):return (V/n).read_text(encoding='utf8')
def put(n,s):
 p=R/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(s,encoding='utf8',newline='\n')
def replace(s,a,b):
 assert a in s,a;return s.replace(a,b)
for n in ['pbd/__init__.py','pbd/models.py','remote.py','check_progress.py','publish.py']:
 (R/n).write_bytes((V/n).read_bytes())
put('publish.py',source('publish.py').replace('readout_patch_v2_20261023','readout_budget_20261023'))
put('PROTOCOL.md',(V/'NEXT_BUDGET_PROTOCOL.md').read_text()+'''\n## Execution binding\nThis new implementation instantiates the above registered contrast. Only P0/G/S,30 trajectories200 epochs, budgets50/100/200; validation-best selection uses earliest strict minimum. Models copied byte-for-byte from v2. Parent patch caches reused read-only in place with actual SHA verification. Each budget50 test intervention and partition must reproduce v2 predictions at atol/rtol1e-5. No periodic model selection by test. Admission9 isolated optimizer calls,0 environments, separate from49800 formal updates. A finite failure stops; no automatic scientific retry, B or additional variants. Full epoch recovery checkpoint preserves model/optimizer/best/selection/RNG/grads/modes/requires-grad/cursor and parent patch binding; no automatic restart.\n''')
bindings={}
for n in ['NEXT_BUDGET_PROTOCOL.md','final_A_SPATIAL_20261023/LOCAL_FINAL_AUDIT.json','final_A_SPATIAL_20261023/CLOSURE_INDEX.json','final_A_SPATIAL_20261023/PAIRED_COMPARISON_AND_GENERALIZATION.json']:
 p=V/n;bindings[str(p.resolve())]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
put('PARENT_LOCAL_ACCEPTANCE.json',json.dumps(dict(status='V2_COMPLETE_SCIENTIFIC_A_FAILED',files=bindings),indent=2))
s=source('pbd/common.py');s=replace(s,"SCALE=torch.tensor",f"V2=Path({VQ!r})\nSCALE=torch.tensor")
s=replace(s,"assert verify_old()==c['old_contract_sha256'];return", "assert verify_old()==c['old_contract_sha256']\n assert sha(V2/'CONTRACT.json')==c['v2_contract_sha256']\n vc=js(V2/'CONTRACT.json');assert all(sha(p)==h for p,h in vc['files'].items())\n return")
put('pbd/common.py',s)
s=source('pbd/admit.py').replace("choices=['readout','spatial']","choices=['budget']").replace("['P0','M','C','Q'] if a.phase=='readout' else ['G','S']","['P0','G','S']")
put('pbd/admit.py',s)
s=source('pbd/train.py').replace("choices=['readout','spatial']","choices=['budget']").replace("['P0','M','C','Q'] if a.phase=='readout' else ['G','S']","['P0','G','S']")
s=s.replace("a.phase=='spatial'","a.phase=='budget'").replace("pf=OUT/'private'/('patch%d.pth'%task)","pf=V2/'output/private'/('patch%d.pth'%task)").replace("OUT/'PATCH_MANIFEST.json'","V2/'output/PATCH_MANIFEST.json'")
s=s.replace('budgets=[50]','budgets=[50,100,200]').replace('range(50)','range(200)').replace('epochs=50','epochs=200')
start=s.index("       if budget==50 and kind in ('P0','M')")
end=s.index("     pf=OUT/'private/predictions'",start)
s=s[:start]+'''       if budget==50:
        oldpf=V2/'output/private/predictions'/(key+'_b50.pth');oldcase=js(V2/'output/cases'/(key+'_b50.json'));assert sha(oldpf)==oldcase['predictions_sha256']
        op=torch.load(oldpf,map_location='cpu')[part];assert torch.equal(ii,op['ids']);ref=op['predictions'][mode]
        parity=dict(max_abs_prediction_m=float((pred-ref).abs().max()),allclose=bool(torch.allclose(pred,ref,atol=1e-5,rtol=1e-5)),parent_predictions_sha256=sha(oldpf))
        case.setdefault('v2_50_epoch_parity',{}).setdefault(part,{})[mode]=parity;assert parity['allclose'],'Unexpected 50-epoch prefix reproduction mismatch'
'''+s[end:]
put('pbd/train.py',s)
s=source('pbd/audit.py').replace("assert phase in ('readout','spatial')","assert phase=='budget'")
s=s.replace("([('READOUT',40)] if phase=='readout' else [('READOUT',40),('SPATIAL',20)])","[('BUDGET',30)]").replace("(2000 if phase=='readout' else 3000)","6000\n assert sum(x['updates'] for x in allrows)==49800")
s=s.replace("(('P0','M','C','Q') if phase=='readout' else ('P0','M','C','Q','G','S'))","('P0','G','S')").replace('for budget in [50]:','for budget in [50,100,200]:')
s=s.replace("(('M','C','Q') if phase=='readout' else ('M','C','Q','G','S'))","('G','S')").replace('b50_task','b200_task')
put('pbd/audit.py',s)
s=source('controller.py').replace('int(3.3*1024**3)','int(.5*1024**3)')
start=s.index(" run('readout_admission'");end=s.index(" write(OUT/'A_GATE.json'",start)
s=s[:start]+" run('budget_admission',['pbd.admit','budget']);run('budget',['pbd.train','budget']);run('budget_audit',['pbd.audit','budget'])\n gate=js(OUT/'BUDGET_GATE.json')\n"+s[end:]
put('controller.py',s)
put('audit_progress.py',source('audit_progress.py').replace('len(rows)<=50','len(rows)<=200').replace('full_50epoch_fits','full_200epoch_fits').replace('len(v)==50','len(v)==200'))
s=source('deploy.py').replace(VQ,Q)
s=s.replace("[R/'PROTOCOL.md',R/'controller.py',", "[R/'PROTOCOL.md',R/'PARENT_LOCAL_ACCEPTANCE.json',R/'controller.py',")
start=s.index("C=Path('/data/mlr_ifa/experiment_workspaces/PHYSICAL_BOTTLENECK")
end=s.index("OLD=Path(",start)
s=s[:start]+'''V2=Path('/data/mlr_ifa/experiment_workspaces/READOUT_PATCH_OBJECT41_20261023_v2')
assert sha(V2/'CONTRACT.json')=='a5667bde73769a1385c05fd1989d839a7e1cb30ab76853e225c579436d0c65d1'
vc=json.loads((V2/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in vc['files'].items())
assert (V2/'controller.exit').read_text().strip()=='0' and all(p.read_text().strip()=='0' for p in V2.glob('*.exit'))
assert sha(V2/'pbd/models.py')==sha(Q/'pbd/models.py')
assert sha(V2/'output/PATCH_MANIFEST.json')=='8344cac6065e3ab190d7343c5f9d35e51eeb81226449b35f181ddcb29389fcee'
for name,h in [('patch6.pth','ea97a9689454f3feccd042c3268ff12ed554634a3ade2853a11b4490ff6d3d3b'),('patch7.pth','2a9b61d9fb0e46039876b3cb83fbbf20d20e9c8bdc3ab54e8715a36334bfa5fa')]:assert sha(V2/'output/private'/name)==h
for p in [V2/'CONTRACT.json',V2/'controller.exit',V2/'output/PATCH_MANIFEST.json',V2/'output/A_GATE.json']:files[str(p)]=sha(p)
for p in (V2/'output/cases').glob('*.json'):files[str(p)]=sha(p)
for pp in Path('/proc').iterdir():
 if pp.name.isdigit():
  try:envbytes=(pp/'environ').read_bytes().split(b'\\\\0')
  except OSError:continue
  assert ('PBD_ROOT='+str(V2)).encode() not in envbytes,'Parent process remains'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
ds=os.statvfs('/data');ss=os.statvfs('/');assert ds.f_bfree*ds.f_frsize>=int(2.5*1024**3) and ss.f_bavail*ss.f_frsize>=1024**3
'''+s[end:]
s=s.replace("version='READOUT_PATCH_A_v2'","version='READOUT_BUDGET_A_v1',v2_contract_sha256=sha(V2/'CONTRACT.json')").replace("scope='A1 existing-checkpoint scalar diagnosis -> A2 matched fusion -> conditional A3 patch; B only after independent A acceptance'","scope='Only training budget 200; fixed P0/G/S;50/100 descriptive,200 gate; no B'")
put('deploy.py',s)
put('IMPLEMENTATION_NOTES.md','''# Budget-only implementation
New source/root, no edits to frozen v2. Models are byte-for-byte v2. Only formal epoch budget is changed; selection remains validation-only with earliest minimum. P0/G/S micro128/16/16, effective128, FP32, AdamW3e-4 wd.01 unchanged. All90 selected CPs saved; all30 trajectories must complete. All50-prefix train/validation/test interventions compared with v2 within original tolerances; prefix checkpoints use same initialization and train RNG order. Extra checkpoint saves do not call model or consume RNG.

Parent accepted closure and preregistration hashes bound by PARENT_LOCAL_ACCEPTANCE.json. Deployment verifies parent source, natural exit0/no own process, GPU idle, private patch SHA and same model source. Parent patch cache read only; no copy/export. Own admission9 updates separately. Full epoch recovery stored; no automatic restart. Private storage budget0.5GiB, reserves2GiB data and1GiB ordinary system. No policy/environment execution. Final auditors must be adapted and actually exercised before acceptance; never inherit v2 final PASS.
''')
for p in R.rglob('*.py'):ast.parse(p.read_text(encoding='utf8'),filename=str(p),feature_version=(3,8))
assert (R/'pbd/models.py').read_bytes()==(V/'pbd/models.py').read_bytes()
put('STATIC_CHECK.json',json.dumps(dict(status='PASS_AST_PY38_AND_EXACT_MODEL_BYTES_NOT_GPU_ADMISSION',files=len(list(R.rglob('*.py'))),model_sha256=hashlib.sha256((R/'pbd/models.py').read_bytes()).hexdigest()),indent=2))
print('Created new budget implementation; static checks passed; not deployed')
