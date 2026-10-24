"""Prepare external audit wrappers only; does not modify the frozen run."""
from pathlib import Path
import ast
R=Path(__file__).resolve().parent;OLD=R.parent/'20261024_ccbp'
def put(n,s):ast.parse(s);(R/n).write_text(s,encoding='utf8')
s=(OLD/'audit_evaluation_progress.py').read_text().replace('INITIAL_A.json','INITIAL_B.json').replace("'ABCD'","'BCNV'").replace('t in (8,9)','t in (6,7,8,9)').replace('(16-len(done))','(32-len(done))').replace('own_admission_updates=13','own_admission_updates=18')
put('audit_evaluation_progress.py',s)
s=(OLD/'final_readonly.py').read_text().replace("Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_GOAL41_20261024_v1');P=Path('/root/mlr_ifa/experiments/CCBP_GOAL41_20261024_v1');", "Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V2_GOAL41_20261024_v1');P=Path('/root/mlr_ifa/experiments/CCBP_GOAL41_20261024_v1');")
s=s.replace("import numpy as np", "OUTPUT={a:Path('/root/mlr_ifa/experiments/CCBP_V2_GOAL41_20261024_v1' if a in 'BN' else '/data/mlr_ifa/experiments/CCBP_V2_GOAL41_20261024_v1')/a for a in 'BCNV'}\nimport numpy as np")
s=s.replace("for arm in 'ABCD'","for arm in 'BCNV'").replace("P/arm/'stage9.pth'","OUTPUT[arm]/'stage9.pth'").replace("P/arm/'recovery.pth'","OUTPUT[arm]/'recovery.pth'")
s=s.replace("teacher=model(O/", "weights=js(Q/'WEIGHTS.json'); weight_by_sha={r['sample_sha256']:r['weight'] for r in weights['rows']}\nassert len(weights['rows'])==len(rows)==430\nct_stored=np.array([r['teacher_cost'] for r in weights['rows']]);assert abs(max(1e-8,float(np.median(ct_stored)))-weights['tau'])<1e-14\nassert all(abs(r['weight']-weights['tau']/(weights['tau']+r['teacher_cost']))<1e-7 for r in weights['rows'])\nteacher=model(O/")
s=s.replace("assert rec['next_epoch']==100", "assert (OUTPUT[arm]/'stage9.pth').stat().st_ino==(OUTPUT[arm]/'recovery.pth').stat().st_ino\n assert rec['weights_sha256']==sha(Q/'WEIGHTS.json')\n assert rec['next_epoch']==100")
s=s.replace("out=[]\n for start", "if arm in 'BC':\n  old=torch.load(P/arm/'stage9.pth',map_location='cpu')['state_dict'];assert all(torch.equal(raw[k],old[k]) for k in raw),'Legacy control final tensor mismatch';strict[arm]['legacy_final_tensor_exact']=True\n out=[]\n for start")
s=s.replace("parts=[]\n  for mu", "tm,ts,tl=qq[0];ti=tl.argmax(-1);ii=torch.arange(len(ti));sm,ss,sl=qq[1]\n  anchor=torch.stack([((sm[:,:,a:b]-tm[ii,ti][:,None,a:b]).square()+(ss[:,:,a:b]-ts[ii,ti][:,None,a:b]).square()).mean(-1)/scale[g].square() for g,(a,b) in enumerate([(0,3),(3,6),(6,7)])],-1)\n  parts=[]\n  for mu")
s=s.replace("student_mixture_nll=float(sn[j])))", "student_mixture_nll=float(sn[j]),student_component_anchor_group_distances=anchor[j].tolist(),student_logits=sl[j].tolist(),teacher_logits=tl[j].tolist(),quality_weight=weight_by_sha[m['sha256']]))")
s=s.replace("assert len(records)==160","assert len(records)==320").replace("assert len(seen)==160","assert len(seen)==320").replace("episodes=160","episodes=320")
s=s.replace("scale=scale.tolist(),manifest_sha256", "weights=weights,weights_sha256=sha(Q/'WEIGHTS.json'),scale=scale.tolist(),manifest_sha256")
put('final_readonly.py',s)
print('External final CPU/record wrappers AST checked only, not final acceptance.')
