"""One-time reviewed derivation. Do not run after deployment; no remote mutations here."""
from pathlib import Path
import ast,hashlib,json,datetime
R=Path(__file__).resolve().parent
B=R.parent/'20261023_readout_budget'
Q='/data/mlr_ifa/experiment_workspaces/READOUT_CAPACITY_OBJECT41_20261023_v1'
BQ='/data/mlr_ifa/experiment_workspaces/READOUT_BUDGET_OBJECT41_20261023_v1'
assert not (R/'ACTIVE_ROOT.json').exists()
(R/'pbd').mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s):p.write_text(s,encoding='utf8')
def change(s,a,b):
 assert a in s,a
 return s.replace(a,b)
for name in ['controller.py','deploy.py','check_progress.py','audit_progress.py','remote.py','publish.py','final_scalars_readonly.py','audit_final_local.py','pbd/__init__.py','pbd/common.py','pbd/models.py','pbd/admit.py','pbd/train.py','pbd/audit.py']:
 s=(B/name).read_text(encoding='utf8')
 # Phase and artifact names only; numerical budget variables are unchanged.
 s=s.replace('READOUT_BUDGET_OBJECT41_20261023_v1','READOUT_CAPACITY_OBJECT41_20261023_v1').replace('READOUT_BUDGET_A_v1','READOUT_CAPACITY_A_v1').replace('BUDGET_', 'CAPACITY_').replace("'BUDGET'","'CAPACITY'").replace("'budget'","'capacity'").replace("'budget_admission'","'capacity_admission'").replace("'budget_audit'","'capacity_audit'")
 # Restore case/checkpoint numerical budget dictionary fields (not phase strings).
 s=s.replace("case['capacity']","case['budget']").replace("st['capacity']","st['budget']")
 s=s.replace("x['capacity']","x['budget']")
 write(R/name,s)
base=R.parent/'20261023_physical_effect_probe/physical_effect_probe/readout_models.py'
s=base.read_text(encoding='utf8')
s=change(s,"def __init__(self,kind='P3'):","def __init__(self,kind='P3',temporal_ffn=128):")
s=change(s,'self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,512,','self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,temporal_ffn,')
write(R/'pbd/readout_base.py',s)
s=(R/'pbd/models.py').read_text()
s=change(s,'from physical_effect_probe.readout_models import LightReadout','from .readout_base import LightReadout')
s=change(s,'def __init__(self,kind):','def __init__(self,kind,temporal_ffn=128):')
s=change(s,"super().__init__('P3');","super().__init__('P3',temporal_ffn=temporal_ffn);")
s=change(s,'def __init__(self):','def __init__(self,temporal_ffn=128):')
s=change(s,"super().__init__('Q');","super().__init__('Q',temporal_ffn=temporal_ffn);")
s=change(s,'def build(kind):','def build(kind,temporal_ffn=128):')
s=change(s,"LightReadout('P0' if kind=='P0' else 'P3')","LightReadout('P0' if kind=='P0' else 'P3',temporal_ffn=temporal_ffn)")
s=change(s,'return FusionReadout(kind)','return FusionReadout(kind,temporal_ffn=temporal_ffn)')
s=change(s,'return SpatialReadout()','return SpatialReadout(temporal_ffn=temporal_ffn)')
write(R/'pbd/models.py',s)
s=(R/'pbd/train.py').read_text()
start=s.index('       if budget==50:');end=s.index("     pf=OUT/'private/predictions'",start)
s=s[:start]+s[end:]
write(R/'pbd/train.py',s)
s=(R/'final_scalars_readonly.py').read_text()
start=s.index('    if budget==50:');end=s.index("    ii.append",start)
s=s[:start]+s[end:]
s=change(s,'model=build(kind);model.load_state_dict',"model=build(kind);assert sum(p.numel() for p in model.parameters())==({'P0':483335,'G':400775,'S':400775}[kind]);model.load_state_dict")
write(R/'final_scalars_readonly.py',s)
s=(R/'pbd/common.py').read_text()
s=change(s,'SCALE=torch.tensor',f"BUDGET=Path({BQ!r})\nSCALE=torch.tensor")
s=change(s," return sha(ROOT/'CONTRACT.json')"," assert sha(BUDGET/'CONTRACT.json')==c['budget_contract_sha256']\n bc=js(BUDGET/'CONTRACT.json');assert all(sha(p)==h for p,h in bc['files'].items())\n return sha(ROOT/'CONTRACT.json')")
write(R/'pbd/common.py',s)
s=(R/'pbd/admit.py').read_text()
insert='''
 # Untrained parameterization proof: same seed, complete state, RNG consumption, output.
 import importlib.util
 spec=importlib.util.spec_from_file_location('pbd.reference_v2_models',str(V2/'pbd/models.py'));reference_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference_module)
 equivalence=[]
 for k in kinds:
  seed(647);old=reference_module.build(k).cuda();old_rng=torch.get_rng_state().clone();old_cuda=[r.clone() for r in torch.cuda.get_rng_state_all()]
  seed(647);new512=build(k,temporal_ffn=512).cuda()
  assert torch.equal(old_rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(old_cuda,torch.cuda.get_rng_state_all()))
  assert old.state_dict().keys()==new512.state_dict().keys() and all(torch.equal(v,new512.state_dict()[n]) for n,v in old.state_dict().items())
  pa=x[:,:,1:3,None,:].expand(-1,-1,-1,196,-1).clone() if k=='S' else None
  with torch.no_grad():
   a=old(x,pa) if k in ('G','S') else old(x);b=new512(x,pa) if k in ('G','S') else new512(x)
  assert torch.equal(a,b)
  equivalence.append(dict(kind=k,state_exact=True,rng_exact=True,output_exact=True,parameters=sum(p.numel() for p in old.parameters()),untrained_forwards=2))
  del old,new512,pa
'''
s=change(s," kinds=['P0','G','S']"," kinds=['P0','G','S']"+insert)
s=change(s,"seed(647);m=build(kind).cuda();opt=", "seed(647);m=build(kind).cuda();assert sum(p.numel() for p in m.parameters())==({'P0':483335,'G':400775,'S':400775}[kind]);opt=")
s=change(s,"rows=rows,optimizer_calls=", "rows=rows,baseline_512_equivalence=equivalence,baseline_untrained_forwards=6,temporal_ffn=128,optimizer_calls=")
write(R/'pbd/admit.py',s)
s=(R/'controller.py').read_text().replace('capacity(int(.5*1024**3))','capacity(int(.4*1024**3))').replace('A_FAILED_PENDING_LOCAL_AUDIT_GENERALIZATION_PROTOCOL','A_FAILED_PENDING_LOCAL_AUDIT_FINAL_DIAGNOSTIC_STOP')
write(R/'controller.py',s)
s=(R/'deploy.py').read_text()
s=change(s,"assert sha(V2/'pbd/models.py')==sha(Q/'pbd/models.py')",f"""BUDGET=Path({BQ!r})
assert sha(BUDGET/'CONTRACT.json')=='ff2e5256594ddbe65ae32e1fdbcbb12cb7ae70951a164f7b47d5ce29792b02bf'
bc=json.loads((BUDGET/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in bc['files'].items())
assert (BUDGET/'controller.exit').read_text().strip()=='0' and all(p.read_text().strip()=='0' for p in BUDGET.glob('*.exit'))
for p in [BUDGET/'CONTRACT.json',BUDGET/'controller.exit',BUDGET/'output/A_GATE.json',BUDGET/'output/BUDGET_RESULTS.json',BUDGET/'output/BUDGET_COST.json',BUDGET/'output/DEMO_SPLITS.json']:
 if p.exists():files[str(p)]=sha(p)
for p in (BUDGET/'output/cases').glob('*.json'):files[str(p)]=sha(p)
assert len(list((BUDGET/'output/cases').glob('*.json')))==90""")
s=change(s,"assert ('PBD_ROOT='+str(V2)).encode() not in envbytes,'Parent process remains'","assert all(('PBD_ROOT='+str(parent)).encode() not in envbytes for parent in (V2,BUDGET)),'Parent process remains'")
s=s.replace('int(2.5*1024**3)','int(2.4*1024**3)')
s=change(s,"v2_contract_sha256=sha(V2/'CONTRACT.json'),files=files", "v2_contract_sha256=sha(V2/'CONTRACT.json'),budget_contract_sha256=sha(BUDGET/'CONTRACT.json'),files=files")
s=change(s,"Only training budget 200; fixed P0/G/S;50/100 descriptive,200 gate; no B", "Only Temporal FFN 512 to128; fixed P0/G/S200epochs;50/100 descriptive,200 gate; final automatic capacity value; no B")
write(R/'deploy.py',s)
s=(R/'check_progress.py').read_text()
s=change(s,"files={}\nfor p in O.iterdir():",f"BUDGET=Path({BQ!r});bc=json.loads((BUDGET/'CONTRACT.json').read_text());bad += [p for p,h in bc['files'].items() if sha(p)!=h]\nif sha(BUDGET/'CONTRACT.json')!=contract['budget_contract_sha256']:bad.append('budget_contract')\nfiles={{}}\nfor p in O.iterdir():")
write(R/'check_progress.py',s)
s=(R/'audit_progress.py').read_text()
s=change(s," assert [r['kind']", " assert a['baseline_untrained_forwards']==6 and a['temporal_ffn']==128 and all(r['state_exact'] and r['output_exact'] and r['rng_exact'] for r in a['baseline_512_equivalence'])\n assert [r['parameters'] for r in a['rows']]==[483335,400775,400775]\n assert [r['kind']")
write(R/'audit_progress.py',s)
s=(R/'publish.py').read_text().replace('reports/readout_budget_20261023','reports/readout_capacity_20261023');write(R/'publish.py',s)
write(R/'PROTOCOL.md',(B/'NEXT_CAPACITY_PROTOCOL.md').read_text(encoding='utf8')+'\n\n实施登记：本目录为容量对照，新源待本源GPU准入；此前登记时间与失败结论保留。\n')
paths=[B/'final_A_BUDGET_20261023'/n for n in ['LOCAL_FINAL_AUDIT.json','BUDGET_DIAGNOSTICS.json','CLOSURE_INDEX.json']]+[B/'NEXT_CAPACITY_PROTOCOL.md',B/'final_scalars_readonly_20261023_171248/EVIDENCE.json',B/'snapshot_20261023_171206/EVIDENCE.json']
for p in paths:assert p.is_file()
closure=json.loads(paths[2].read_text());assert closure['status']=='PASS_COMPLETE_LOCAL_CLOSURE' and closure['selected'] is None
assert closure['final_audit_sha256']==sha(paths[0]) and closure['diagnostics_sha256']==sha(paths[1])
write(R/'PARENT_LOCAL_ACCEPTANCE.json',json.dumps(dict(status='BOUND_ACCEPTED_BUDGET_BASELINE',budget_contract_sha256='ff2e5256594ddbe65ae32e1fdbcbb12cb7ae70951a164f7b47d5ce29792b02bf',files={str(p.resolve()):sha(p) for p in paths},old_scientific_failure_preserved=True),indent=2))
checked=[]
for p in R.rglob('*.py'):ast.parse(p.read_text(encoding='utf8'),feature_version=(3,8));checked.append(p.relative_to(R).as_posix())
write(R/'STATIC_CHECK.json',json.dumps(dict(status='PASS_AST_ONLY_NOT_GPU',python38_files=checked,baseline_model_source_sha256=sha(base),expected_parameters=dict(P0=483335,G=400775,S=400775)),indent=2))
write(R/'RUN_STATUS.json',json.dumps(dict(status='PARTIAL',phase='IMPLEMENTED_NOT_DEPLOYED',observation=datetime.datetime.now().astimezone().isoformat(),policy_updates=0,environment_episodes=0),indent=2))
print(json.dumps(dict(status='DERIVED_STATIC_ONLY',files=len(checked),root=str(R))))
