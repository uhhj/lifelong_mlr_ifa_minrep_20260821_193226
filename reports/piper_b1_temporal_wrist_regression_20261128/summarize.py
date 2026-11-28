"""Recompute region metrics and split observable errors from ungradable closure frames."""
from pathlib import Path
from statistics import median
from collections import Counter
import hashlib,json,sys
def summary(folder):
    read=lambda name:json.loads((folder/name).read_bytes())
    audit=read('audit.json');labels=read('labels.json')
    out={}
    def dual(rs,key):
        return [round(median([r[key][i] for r in rs]),3) for i in (0,1)] if rs else [None,None]
    for g in ('regression','archive'):
        data=read(g+'.json');eps=data['episodes'];dirs=data['directions']
        assert len(eps)==(32 if g=='regression' else 20)
        assert len(dirs)==2*len(data['pairs'])
        index={(r['recipient'],r['donor']):r['correction_mm'] for r in dirs}
        pair=[[.5*(index[a,b][i]+index[b,a][i]) for i in (0,1)] for a,b in data['pairs']]
        byregion={}
        for region in ('左上','右上','左下','右下','全部'):
            rs=[(ep,r) for ep,r in eps.items() if region=='全部' or r['region']==region]
            ds=[r for r in dirs if region=='全部' or r['region']==region]
            closes=[Counter(),Counter()]
            for ep,r in rs:
                for i,event in enumerate(r['close_events']):
                    if event is None:closes[i]['miss']+=1;continue
                    token=hashlib.sha256((ep+str(event['frame'])).encode()).hexdigest()[:8]
                    assert labels[token] in ('aligned','uncertain','outside')
                    category='outside' if event['delta_z_to_expert_mm']>35 else labels[token]
                    closes[i][category]+=1
            byregion[region]=dict(episodes=len(rs),directions=len(ds),natural_mm=dual([r for _,r in rs],'natural_mm'),
                swap_mm=dual(ds,'correction_mm'),closes=closes,active=sum(r['active'] for _,r in rs),
                view=Counter(str(r['view']) for _,r in rs))
        out[g]=dict(regions=byregion,paired_mm=[round(median([r[i] for r in pair]),6) for i in (0,1)],
                    positive_pairs=[sum(r[i]>0 for r in pair) for i in (0,1)],pairs=len(pair))
    reg=audit['registration']
    metrics=['early_px','late_static_px','runtime_old_mm','manual_runtime_mm','early_new_map_mm',
             'old_px_new_map_mm','old_map_mm','contact_ref_difference_mm']
    out['registration']={k:round(median([r[k] for r in reg if r[k] is not None]),3) for k in metrics}
    out['controls']=[r['versions'][1]['simulated_width_goals'] for r in audit['field']]
    return out
if __name__=='__main__':
    folder=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
    print(json.dumps(summary(folder),ensure_ascii=False,indent=2))
