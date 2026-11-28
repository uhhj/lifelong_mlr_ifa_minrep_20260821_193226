"""Recompute fixed-window, continuous-history, and first-command outcomes."""
from collections import Counter
from statistics import median
from pathlib import Path
import hashlib,json,sys
def read(p):return json.loads(p.read_bytes())
def summarize(folder):
    labels=read(folder/'labels.json')
    result={}
    def pair_medians(values):
        return [round(median(p[i] for p in values),6) for i in (0,1)] if values else [None,None]
    for g,n in [('regression',32),('archive',20)]:
        j=read(folder/(g+'.json'));long=read(folder/('continuous_'+g+'.json'));eps=j['episodes']
        assert len(eps)==n and set(eps)==set(long)
        directions={(r['recipient'],r['donor']):r['correction_mm'] for r in j['directions']}
        assert len(directions)==2*len(j['pairs'])
        pair=[[.5*(directions[a,b][i]+directions[b,a][i]) for i in (0,1)] for a,b in j['pairs']]
        group={}
        for zone in ('左上','右上','左下','右下','全部'):
            ids=[e for e,r in eps.items() if zone=='全部' or zone==r['region']]
            ds=[r['correction_mm'] for r in j['directions'] if zone=='全部' or zone==r['region']]
            counts=[Counter(),Counter()]
            for e in ids:
                for i,evt in enumerate(eps[e]['close_events']):
                    if not evt:counts[i]['miss']+=1;continue
                    token=hashlib.sha256((e+str(evt['frame'])).encode()).hexdigest()[:8]
                    c=labels[token];assert c in ('aligned','outside','uncertain')
                    counts[i]['outside' if evt['delta_z_to_expert_mm']>35 else c]+=1
            short=[eps[e]['natural_mm'] for e in ids];continuous=[long[e]['natural'] for e in ids]
            group[zone]=dict(n=len(ids),directions=len(ds),swapped=pair_medians(ds),fixed=pair_medians(short),
              continuous=pair_medians(continuous),positive_fixed=[sum(p[i]>.001 for p in short) for i in (0,1)],
              positive_continuous=[sum(p[i]>.001 for p in continuous) for i in (0,1)],
              unchanged_fixed=[sum(abs(p[i])<=.001 for p in short) for i in (0,1)],
              unchanged_continuous=[sum(abs(p[i])<=.001 for p in continuous) for i in (0,1)],closure=counts)
        result[g]=dict(regions=group,paired=pair_medians(pair),positive_pairs=[sum(p[i]>0 for p in pair) for i in (0,1)])
    return result
if __name__=='__main__':print(json.dumps(summarize(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent),ensure_ascii=False,indent=2))
