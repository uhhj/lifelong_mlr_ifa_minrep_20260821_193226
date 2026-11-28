"""Independently verify window identity, paired physical errors, primary counts and reporting coverage."""
import json,hashlib,math,re
from statistics import median
from pathlib import Path
here=Path(__file__).resolve().parent
prior=here.parent/'displacement_audit_1128'
def read(p):return json.loads(p.read_bytes())
def verify():
    s=read(here/'summary.json');out=read(here/'episode_outcomes.json');all_eps=set();cases=[];frozen=[]
    for group,want in [('regression',32),('archive',20)]:
        pair=read(here/('paired_'+group+'.json'))
        prev=read(prior/('windows_'+group+'.json'))
        assert len(pair)==want and prev.keys()==pair.keys() and not set(pair)&all_eps
        all_eps.update(pair)
        for e,row in pair.items():
            assert e in out and not row['in_train']
            first={r['t']:r for r in prev[e]['windows']}
            assert row['source_file_stamp_ns']<=500 and len(first)==len(row['windows'])
            for w in row['windows']:
                old=first[w['t']];assert (w['grid'],w['phase'])==(old['grid'],old['phase'])
                vs=w['variants'];v=vs['train_zero'];p=w['station']
                assert abs(math.dist(p,v['end'])-v['tip8'])<.000004
                assert 0<=v['tip8']<=v['tip_max']+.0000001 and math.dist(p,old['raw'])<.003
                frozen.append(v['tip8'])
                if w['grid']:cases.append((old,w))
    assert len(all_eps)==52 and set(out)==all_eps and len(cases)==s['grid']==444 and len(frozen)==s['windows']==3384
    for ph,key,n in [('correct','correct',183),('descend','descend',29)]:
        sub=[x for x in cases if x[0]['phase']==ph];assert len(sub)==n
        for k in ['station','train_zero','component_a','component_b','aug_a','aug_b']:
            score=[]
            for old,w in sub:
                e=old['expert'];v=w['station'] if k=='station' else w['variants'][k]['end']
                score.append((e[0]*v[0]+e[1]*v[1])/math.hypot(*e[:2]) if ph=='correct' else v[2])
            expect=s['phase']['all'][ph]['versions'][k]
            assert abs(median(score)-expect['median'])<1e-6
            assert sum((x<0 if ph=='correct' else x>=0) for x in score)==expect['wrong']
    z=s['modes']['all']['train_zero']['tip8'];assert abs(max(frozen)-z['max'])<1e-7 and abs(median(frozen)-z['p50'])<1e-7
    training=read(here/'training_control.json')
    assert not set(training['episodes'])&all_eps and len(training['episodes'])==50 and sum(e['rows'] for e in training['episodes'].values())==9731
    comp=read(here/'components.json')
    assert sum(len(ws) for ws in comp.values())==212
    assert all(len(w['weights'])==12 and abs(sum(w['weights'])-1)<1e-7 for ws in comp.values() for w in ws)
    html=(here/'B1_双路径逐窗核对.html').read_text(encoding='utf8')
    payload=json.loads(re.search(r'const rows=(\[.*?\]);const \$',html).group(1))
    assert len(payload)==3384 and {(r[0],r[1]) for r in payload}=={(e,r['t']) for group in ('regression','archive') for e,ep in read(here/('paired_'+group+'.json')).items() for r in ep['windows']}
    print('verified 52 holdouts, 50 source-cache controls, 444 disjoint windows, 212 full mixtures, report and viewer')
if __name__=='__main__':verify()
