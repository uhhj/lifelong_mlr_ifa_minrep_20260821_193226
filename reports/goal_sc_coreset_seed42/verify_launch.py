"""Local metadata-only verification, not a substitute for actual GPU evidence."""
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import json,hashlib
R=Path(__file__).resolve().parent;E=R/'launch_evidence'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    package=js(E/'PACKAGE_MANIFEST.json')
    for n,r in package['files'].items():assert (E/n).stat().st_size==r['bytes'] and sha(E/n)==r['sha256'],n
    contract=js(E/'CONTRACT.json')
    for p in (R/'source').rglob('*'):
        if p.is_file() and p.suffix in {'.py','.yaml','.sh'} and '__pycache__' not in p.parts:
            assert sha(p)==contract['source_sha256'][p.relative_to(R/'source').as_posix()],str(p)
    pre=js(E/'GPU_PREFLIGHT.json');assert pre['status']=='PASS' and pre['isolated_optimizer_calls']==4 and pre['environment_steps']==0
    assert (E/'software.exit').read_text().strip()=='0' and (E/'preflight.exit').read_text().strip()=='0'
    summary=[]
    for t in range(6,10):
        b=js(E/'binding_metadata'/f'task{t}.json');assert b['last_actions_omitted_for_publication']
        w=b['windows'];report=js(E/'selection_preflight'/f'task{t}.json');n=len(w);k=[499,445,452,985][t-6]
        pop=Counter(str(x['group']) for x in w.values());old=Counter(str(w[x]['group']) for x in report['original_ids']);new=Counter(str(w[x]['group']) for x in report['selected_ids'])
        assert len(set(report['selected_ids']))==k
        assert len(set(report['original_ids'])-set(report['selected_ids']))==report['minimum_swaps']==len(report['swaps'])
        for g,ng in pop.items():
            assert old[g]==report['original'].get(g,0) and new[g]==report['target'][g]
            assert abs(F(new[g],k)-F(ng,n))<F(1,k)
        if report['minimum_swaps']==0:assert report['original_ids']==report['selected_ids']
        for g in pop:
            if old[g]<=report['target'][g]:assert all(x in report['selected_ids'] for x in report['original_ids'] if str(w[x]['group'])==g)
        def profile(c,total):return dict(positive_pct=float(100*F(sum(v for g,v in c.items() if int(g)%3==2),total)),endpoint_pct=float(100*F(sum(v for g,v in c.items() if int(g)//3==4),total)))
        summary.append(dict(task=t,N=n,K=k,swaps=report['minimum_swaps'],full=profile(pop,n),original=profile(old,k),sc=profile(new,k),radius_before=report['radius_before'],radius_after=report['radius_after'],demo_original=report['demo_original'],demo_final=report['demo_final']))
    out=dict(status='PASS_LAUNCH_EVIDENCE',formal_completion=False,files_checked=len(package['files']),selection=summary,progression=dict(AUC='>63/80',OldFinal='>=41/60'),notes='All joint cells checked; marginal percentages are descriptive only.')
    (R/'LOCAL_LAUNCH_ACCEPTANCE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
