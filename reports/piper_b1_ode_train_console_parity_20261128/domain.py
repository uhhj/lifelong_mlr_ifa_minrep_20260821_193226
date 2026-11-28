"""Compare held-out states/anchors to the fixed training normalizer, without changing inputs."""
from inputs import *
def run():
    stats=read(root/'run/protocol.json')['stats']
    result={}
    for group in ('regression','archive'):
        q=read(last/('windows_'+group+'.json'))
        for ep,rec in subjects(group):
            seq=rec['rows'];rows=[]
            for w in q[ep]['windows']:
                t=w['t'];vals={'agent_pos':np.array([rp.at(r,'feedback') for r in seq[t-7:t+1]]),'action':np.array([rp.at(seq[t-1],'target')])}
                spans={}
                for k,x in vals.items():
                    lo=np.array(stats[k]['min']);hi=np.array(stats[k]['max']);span=hi-lo
                    scale=np.where(span<1e-4,1.,2./np.maximum(span,1e-4));offset=np.where(span<1e-4,-lo,-1.-lo*scale)
                    spans[k]=np.max(abs(x*scale+offset),axis=0).round(5).tolist()
                rows.append(dict(t=t,**spans))
            result[ep]=rows
    write('domain.json',dict(episodes=result,limits={k:{m:stats[k][m] for m in ('min','max')} for k in ('agent_pos','action')}))
if __name__=='__main__':run()
