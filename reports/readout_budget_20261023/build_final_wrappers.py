"""Derive external budget-aware independent auditors; no training source edits."""
from pathlib import Path
import ast,json
R=Path(__file__).resolve().parent;V=R.parent/'20261023_readout_patch_v2'
s=(V/'final_scalars_readonly.py').read_text()
s=s.replace('READOUT_PATCH_OBJECT41_20261023_v2','READOUT_BUDGET_OBJECT41_20261023_v1')
s=s.replace("phase='SPATIAL' if (OUT/'SPATIAL_GATE.json').exists() else 'READOUT'","phase='BUDGET'")
s=s.replace("kinds=['P0','M','C','Q']+(['G','S'] if phase=='SPATIAL' else [])","kinds=['P0','G','S']")
s=s.replace("(['READOUT','SPATIAL'] if phase=='SPATIAL' else ['READOUT'])","['BUDGET']")
s=s.replace("assert len(fits)==len(epochs)==len(kinds)*10","assert len(fits)==len(epochs)==30\nassert sum(r['epochs'] for r in fits)==6000 and sum(r['updates'] for r in fits)==49800\nfor r in js(V2/'output/PATCH_MANIFEST.json')['tasks']:\n assert sha(V2/'output/private'/('patch%d.pth'%r['task']))==r['sha256']")
start=s.index('  ii=[];pred=');end=s.index('assert verify()==contract',start)
block=s[start:end]
block=block.replace('list(range(50));best=min(es,','list(range(200));best=min(es[:budget],')
block=block.replace('math.ceil(nt/128)*50','math.ceil(nt/128)*200')
block=block.replace("key+'_b50.json'","key+'_b%d.json'%budget").replace("key+'_b50.pth'","key+'_b%d.pth'%budget")
block=block.replace("assert st['epoch']==best['epoch']==sel['selected_epoch']", "assert st['budget']==budget and case['budget']==budget and st['validation_score_m']==best['validation_position_m']==sel['validation_score_m'];assert st['epoch']==best['epoch']==sel['selected_epoch']")
block=block.replace("   ii.append(pp['test']['ids'])", """   if budget==50:
    oldpf=V2/'output/private/predictions'/(key+'_b50.pth');oldcase=js(V2/'output/cases'/(key+'_b50.json'));assert sha(oldpf)==oldcase['predictions_sha256'];oldpp=torch.load(oldpf,map_location='cpu')
    for part in pp:
     assert torch.equal(pp[part]['ids'],oldpp[part]['ids'])
     for mode,p in pp[part]['predictions'].items():assert torch.allclose(p,oldpp[part]['predictions'][mode],atol=1e-5,rtol=1e-5)
   ii.append(pp['test']['ids'])""")
block=block.replace("cases['%s_task%d'%(kind,task)]=dict(task=task,model=kind,", "cases['%s_b%d_task%d'%(kind,budget,task)]=dict(task=task,model=kind,budget=budget,")
s=s[:start]+'  for budget in (50,100,200):\n'+''.join(' '+line+'\n' for line in block.splitlines())+s[end:]
s=s.replace('assert verify()==contract','assert len(cps)==90 and len(cases)==18\nassert verify()==contract')
(R/'final_scalars_readonly.py').write_text(s,encoding='utf8')
s=(V/'audit_final_local.py').read_text()
s=s.replace("'%s_b50_task%d'%(x['model'],x['task'])","'%s_b%d_task%d'%(x['model'],x['budget'],x['task'])")
s=s.replace("cases[x['task'],x['model']]=x","cases[x['task'],x['model'],x['budget']]=x")
s=s.replace("(('M','C','Q') if manifest['phase']=='READOUT' else ('M','C','Q','G','S'))","('G','S')")
s=s.replace("cases[task,kind];b=cases[task,'P0']","cases[task,kind,200];b=cases[task,'P0',200]")
s=s.replace('checks={}',"assert manifest['phase']=='BUDGET' and len(cases)==18 and len(manifest['checkpoint_sha256'])==90\nassert manifest['actual_cost']['fits']==30 and manifest['actual_cost']['epochs']==6000 and manifest['actual_cost']['updates']==49800\nchecks={}")
(R/'audit_final_local.py').write_text(s,encoding='utf8')
for n in ['final_scalars_readonly.py','audit_final_local.py']:
 ast.parse((R/n).read_text(),filename=n,feature_version=(3,8))
(R/'FINAL_WRAPPER_STATIC_CHECK.json').write_text(json.dumps(dict(status='PASS_AST_ONLY_NOT_EXERCISED',expected_fits=30,expected_CP=90,expected_merged_cases=18,gate_budget=200,formal_updates=49800),indent=2))
print('Final wrappers built; actual execution pending complete run')
