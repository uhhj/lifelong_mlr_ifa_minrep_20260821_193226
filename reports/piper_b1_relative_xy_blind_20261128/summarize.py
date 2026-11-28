"""Aggregate paired direction, real-target approach distance and adjudicated close events."""
from collections import Counter
import hashlib,json,statistics,sys
from pathlib import Path


def med(xs):
    return round(statistics.median(xs),3) if xs else None


def both(rows,key):
    return [med([r[key][i] for r in rows]) for i in range(2)]


def analyze(root):
    read=lambda n:json.loads((root/n).read_bytes())
    labels=read('adjudication.json')
    summaries={}
    for group in ('regression','blind','archive'):
        data=read(group+'.json')
        rows=data['episodes'];dirs=data['directions']
        assert len(dirs)==len(data['pairs'])*2
        idx={(r['recipient'],r['donor']):r for r in dirs}
        pairs=[]
        for a,b in data['pairs']:
            runs=[idx[a,b],idx[b,a]]
            pairs.append([sum(r['correction_mm'][v] for r in runs)/2 for v in (0,1)])
        def info(rows,dirs):
            usable=[r for r in rows if 'natural_mm' in r]
            close=[Counter(),Counter()]
            for r in rows:
                for i,e in enumerate(r['close_events']):
                    if e is None:close[i]['no_command']+=1
                    else:
                        token=hashlib.sha256((r['ep']+str(e['frame'])).encode()).hexdigest()[:8]
                        close[i][labels.get(token,'unreviewed')]+=1
            return dict(episodes=len(rows),scored=len(usable),natural_mm=both(usable,'natural_mm'),
                        single_swaps=len(dirs),swapped_response_mm=both(dirs,'correction_mm'),
                        swapped_advance_mm=both(dirs,'advance_mm'),close=close)
        episodes=[dict(r,ep=e) for e,r in rows.items()]
        reg={k:info([r for r in episodes if r['region']==k],[d for d in dirs if d['region']==k]) for k in ('左上','右上','左下','右下',None)}
        summaries[group]=dict(regions=reg,total=info(episodes,dirs),pair_count=len(pairs),
                pair_response_mm=[med([r[i] for r in pairs]) for i in (0,1)],
                positive_pairs=[sum(r[i]>0 for r in pairs) for i in (0,1)],
                over_1mm_pairs=[sum(r[i]>1 for r in pairs) for i in (0,1)],
                novel_10mm=info([r for r in episodes if (r['nearest_prior_mm'] or 0)>=10],
                                [d for d in dirs if (rows[d['donor']]['nearest_prior_mm'] or 0)>=10]))
    return summaries

if __name__=='__main__':
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
    print(json.dumps(analyze(root),ensure_ascii=False,indent=2))
