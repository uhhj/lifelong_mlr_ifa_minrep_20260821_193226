"""Independent scalar checks for the published aggregates and the private review UI."""
from pathlib import Path
from math import hypot
from statistics import median
from collections import Counter
import json
import re
import hashlib
here=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_bytes())
def verify():
    a=read(here/'aggregate.json');ids=read(here/'identity_check.json');epout=read(here/'episode_outcomes.json')
    assert len(epout)==52 and ids['field']!=ids['candidate']
    counts=Counter();vs={};evidence=[]
    for group in ('regression','archive'):
        ww=read(here/f'windows_{group}.json');rr=read(here/f'rolling_{group}.json')
        micro=read(here/f'micro_{group}.json')
        for ep,subset in micro.items():
            for tick,pred in subset.items():
                row=next(w for w in ww[ep]['windows'] if w['t']==int(tick))
                assert row['grid'] and row['phase']=='other' and 1<=hypot(*row['expert'][:2])<3
        for name in [f'windows_{group}.json',f'rolling_{group}.json',f'micro_{group}.json']:
            assert hashlib.sha256((here/name).read_bytes()).hexdigest()==a['file_sha256'][name]
        for ep,d in ww.items():
            g=[q for q in d['windows'] if q['grid']]
            assert all(y['t']-x['t']>=8 for x,y in zip(g,g[1:]))
            for w in g:
                e=w['expert'];p=w['phase'];counts[p]+=1
                if p not in ('correct','descend'):continue
                roll=rr[ep][str(w['t'])]
                for n in ('field','candidate'):
                    assert abs(w[n]['plan'][2]-w['raw'][2])<=.00101
                    v=roll[n]['final'];assert (v is None)==(roll[n]['guard'] is not None)
                    if v is not None:
                        proj=(v[0]*e[0]+v[1]*e[1])/hypot(*e[:2]) if hypot(*e[:2]) else 0.
                        vs.setdefault((p,n),[]).append((proj,hypot(*v[:2]),v[2]))
                    assert roll[n]['initial_check_mm']<.003
            assert {w['t'] for w in d['windows'] if w['phase']!='other' or w['t']==d['close']-18}==set(map(int,rr[ep]))
            evidence.append(ep)
    assert len(evidence)==len(set(evidence))==52 and dict(counts)==a['counts']['both']['grid_phase']
    for (phase,version),values in vs.items():
        fin=a['groups']['both']['phases'][phase][version]['stages']['final']
        assert len(values)==fin['n']
        for ix,key in enumerate(('projection','xy','z')):
            assert abs(median(x[ix] for x in values)-fin[key])<=.000051
    raw=(here/'B1_八步逐窗对照.html').read_text(encoding='utf8')
    data=json.loads(re.search(r'const data=(\[.*?\]);const labels',raw).group(1))
    assert len(data)==a['counts']['both']['windows'] and len(set((r[0],r[1]) for r in data))==len(data)
    assert len({r[0] for r in data if r[4]})==52
    assert sum(r[13] for r in data)==len(a['exploratory_micro'])
    report=(here/'B1_52条逐条对照.md').read_text(encoding='utf8')
    assert all(report.count('## '+ep+'\n')==1 for ep in epout)
    print('verified group totals, full rolling coverage, six-axis height invariance, all final-vector medians, viewer and 52 episode entries')
if __name__=='__main__':verify()
