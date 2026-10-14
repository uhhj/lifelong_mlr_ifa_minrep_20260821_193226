import argparse,collections,time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from .common import *

def dist(f,ii,jj,start,stop,scale):
    out=np.empty(len(ii))
    for k in range(0,len(ii),4096):
        a=(f[ii[k:k+4096],start:stop].astype(float)-f[jj[k:k+4096],start:stop].astype(float))/scale
        out[k:k+4096]=np.sqrt(np.mean(a*a,axis=(1,2)))
    return out

def phys(d,ii,jj):
    e=np.linalg.norm(d['eef'][ii]-d['eef'][jj],axis=1)
    qi=normalized(d['quat'][ii]);qj=normalized(d['quat'][jj]);rot=np.degrees(2*np.arccos(np.minimum(1,np.abs((qi*qj).sum(1)))))
    gi=(d['state'][ii,7]-d['state'][ii,8])/.08;gj=(d['state'][jj,7]-d['state'][jj,8])/.08
    grip=np.abs(gi-gj);valid=(gi>=0)&(gi<=1)&(gj>=0)&(gj<=1)
    rel=np.linalg.norm(d['target_relative'][ii]-d['target_relative'][jj],axis=1)
    obj_valid=d['target_valid'][ii]&d['target_valid'][jj]
    ok=valid&(e<=.03)&(rot<=15)&(grip<=.1)
    return dict(eef_m=e,rotation_deg=rot,gripper_difference=grip,gripper_in_range=valid,physical_pass=ok,object_relative_error_m=rel,object_available=obj_valid,object_proxy_pass=ok&obj_valid&(rel<=.05))

def select(rows,k):
    # Unique trajectory pairs first. Order independent of score/loss/benefit.
    rows=sorted(rows,key=lambda r:hashlib.sha256((r['new_file']+'|'+r['old_file']).encode()).hexdigest())
    selected=[];seen=set()
    for r in rows:
        key=(r['new_demo'],r['old_task'],r['old_demo'])
        if key not in seen:selected.append(r);seen.add(key)
        if len(selected)==k:return selected
    for r in rows:
        if r not in selected:selected.append(r)
        if len(selected)==k:return selected
    return selected

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();t0=time.time()
    pilot=a.repo/'reports/abcr_object_pilots_v1';dm=read(pilot/'DATA_MANIFEST.json');norm=read(pilot/'NORMALIZATION.json');th=read(pilot/'FROZEN_STAGE_THRESHOLDS.json')
    binding={p.name:sha(p) for p in [a.out/'PROTOCOL.json',a.out/'WINDOW_IDENTITY_MANIFEST.json',a.out/'STAGE_MANIFEST.json',Path(__file__),Path(__file__).with_name('common.py')]}
    write(a.out/'ANALYSIS_BINDING.json',binding,True)
    arrays={};meta=[];keys={};offset=0
    for r in dm['demos']:
        if r['task']<6:continue
        rp=pilot/(r['id']+'_raw.npz');fp=pilot/(r['id']+'_features.npz')
        assert sha(rp)==r['raw_sha256'] and sha(fp)==read(pilot/(r['id']+'_FEATURE_COMPLETE.json'))['sha256']
        z=np.load(rp);f=np.load(fp)
        for key in ['state','eef','quat','target_relative','target_valid']:arrays.setdefault(key,[]).append(z[key])
        arrays.setdefault('visual',[]).append(f['visual']);future=np.zeros((r['n'],8,7),np.float32);valid8=np.arange(r['n'])+8<r['n'];actions=z['actions']
        for h in range(8):
            n=max(0,r['n']-h-1);future[:n,h]=actions[h+1:]
        arrays.setdefault('future',[]).append(future);arrays.setdefault('valid8',[]).append(valid8);arrays.setdefault('valid4',[]).append(np.arange(r['n'])+4<r['n'])
        for ti in range(r['n']):keys[(r['task'],r['demo'],ti)]=offset+ti;meta.append(dict(task=r['task'],demo=r['demo'],t=ti,hdf5=r['hdf5'],language=r['language']))
        offset+=r['n']
    d={k:np.concatenate(v) for k,v in arrays.items()};del arrays
    va=normalized(d['visual'][:,0].astype(float));ve=normalized(d['visual'][:,1].astype(float));state=(d['state']-norm['state_mean'])/norm['state_std'];scale=np.array(norm['action_scale'])
    mapping=read(a.out/'WINDOW_IDENTITY_MANIFEST.json')['records'];lookup={(r['seed'],r['task'],r['record']):r for r in mapping};r1=[];r2=[];r3=[];traj=[];selected=[];byold=[]
    for s in read(a.out/'STAGE_MANIFEST.json')['stages']:
        seed,task=s['seed'],s['task'];dest=a.out/('seed%d_task%d'%(seed,task));dest.mkdir(exist_ok=True)
        new_all=[r for r in mapping if r['seed']==seed and r['task']==task];old_all=[]
        for name in s['old_files']:
            t,record=name[1:].split('__',1);old_all.append(lookup[(seed,int(t),record)])
        nr=[r for r in new_all if r['matches']==1];orr=[r for r in old_all if r['matches']==1]
        ids=lambda rows:np.array([keys[(r['task'],r['demo'],r['anchor_t'])] for r in rows],int)
        ni,oi=ids(nr),ids(orr);cfg=next(t for t in th if t['task']==task and t['beta']==.5)
        score=.5*(va[ni]@va[oi].T+ve[ni]@ve[oi].T);ds=cdist(state[ni],state[oi]);comp=(score>=cfg['tau_v'])&(ds<=cfg['tau_s'])
        il,jl=np.where(comp);ii,jj=ni[il],oi[jl];q4=d['valid4'][ii]&d['valid4'][jj];il,jl,ii,jj=il[q4],jl[q4],ii[q4],jj[q4]
        da=dist(d['future'],ii,jj,0,4,scale);ph=phys(d,ii,jj);share=da<=cfg['tau_plus'];bif=da>=cfg['tau_minus']
        save(dest/'candidate_pairs.npz',new_local=il,old_local=jl,new_global=ii,old_global=jj,action_distance_q4=da,share=share,bif=bif,**ph)
        write(dest/'record_order.json',dict(new=nr,old=orr))
        for gate,ok in [('quantile',np.ones(len(ii),bool)),('physical',ph['physical_pass']),('physical_object_proxy',ph['object_proxy_pass'])]:
            row=dict(seed=seed,task=task,gate=gate,n_current=len(new_all),n_old=len(old_all),n_current_identified=len(nr),n_old_identified=len(orr),n_unique_current_anchors=len(set(ni)),n_unique_old_anchors=len(set(oi)),
                comparable_q4_pairs=int(ok.sum()),share_pairs=int((ok&share).sum()),bif_pairs=int((ok&bif).sum()),
                state_coverage=len(np.unique(il[ok]))/len(new_all),share_coverage=len(np.unique(il[ok&share]))/len(new_all),bif_coverage=len(np.unique(il[ok&bif]))/len(new_all),
                share_unique_trajectory_pairs=len(set((nr[i]['demo'],orr[j]['task'],orr[j]['demo']) for i,j in zip(il[ok&share],jl[ok&share]))),
                bif_unique_trajectory_pairs=len(set((nr[i]['demo'],orr[j]['task'],orr[j]['demo']) for i,j in zip(il[ok&bif],jl[ok&bif]))))
            r1.append(row)
            for ot in sorted(set(r['task'] for r in orr)):
                mask=np.array([orr[j]['task']==ot for j in jl]);byold.append(dict(seed=seed,task=task,gate=gate,old_task=ot,share_pairs=int((ok&share&mask).sum()),bif_pairs=int((ok&bif&mask).sum())))
        for label,mask in [('SHARE',share),('BIF',bif)]:
            r2.append(dict(seed=seed,task=task,label=label,n=int(mask.sum()),physical_pass=int((mask&ph['physical_pass']).sum()),object_proxy_pass=int((mask&ph['object_proxy_pass']).sum()),
                eef_m_median=float(np.median(ph['eef_m'][mask])) if mask.any() else None,rotation_deg_median=float(np.median(ph['rotation_deg'][mask])) if mask.any() else None,
                gripper_difference_median=float(np.median(ph['gripper_difference'][mask])) if mask.any() else None,object_relative_median=float(np.median(ph['object_relative_error_m'][mask])) if mask.any() else None))
            if seed==42:
                for passed,count in [(True,8),(False,4)]:
                    rows=[]
                    for k in np.flatnonzero(mask&(ph['physical_pass']==passed)):
                        n,o=nr[il[k]],orr[jl[k]]
                        rows.append(dict(seed=seed,task=task,label=label,physical_pass=bool(passed),object_proxy_pass=bool(ph['object_proxy_pass'][k]),new_file=n['file'],old_file=o['file'],new_demo=n['demo'],old_task=o['task'],old_demo=o['demo'],new_t=n['anchor_t'],old_t=o['anchor_t'],new_window=n['window_start'],old_window=o['window_start'],new_hdf5=meta[ii[k]]['hdf5'],old_hdf5=meta[jj[k]]['hdf5'],new_language=meta[ii[k]]['language'],old_language=meta[jj[k]]['language'],visual_score=float(score[il[k],jl[k]]),state_distance=float(ds[il[k],jl[k]]),action_distance=float(da[k]),eef_m=float(ph['eef_m'][k]),rotation_deg=float(ph['rotation_deg'][k]),object_relative_m=float(ph['object_relative_error_m'][k])))
                    selected.extend(select(rows,count))
        common=d['valid8'][ii]&d['valid8'][jj];ci,cj=ii[common],jj[common];ds1=dist(d['future'],ci,cj,0,1,scale);ds4=da[common];later=dist(d['future'],ci,cj,4,8,scale)
        for gate,ok in [('quantile',np.ones(len(ci),bool)),('physical',ph['physical_pass'][common])]:
            for q,initial in [(1,ds1),(4,ds4)]:
                for label,initialmask,opposite in [('SHARE',initial<=cfg['tau_plus'],later>=cfg['tau_minus']),('BIF',initial>=cfg['tau_minus'],later<=cfg['tau_plus'])]:
                    chosen=ok&initialmask;n=int(chosen.sum());r3.append(dict(seed=seed,task=task,gate=gate,q=q,label=label,common_q8_pairs=len(ci),q4_pairs=len(ii),excluded_tail_pairs=len(ii)-len(ci),initial_pairs=n,later_opposite_pairs=int((chosen&opposite).sum()),later_opposite_rate=float(opposite[chosen].mean()) if n else None))
                    for demo in range(10):
                        demo_mask=np.array([meta[i]['demo']==demo for i in ci]);sel=chosen&demo_mask
                        traj.append(dict(seed=seed,task=task,gate=gate,q=q,label=label,demo=demo,n=int(sel.sum()),opposite=int((sel&opposite).sum()),rate=float(opposite[sel].mean()) if sel.any() else None))
        save(dest/'nonoverlap.npz',new_global=ci,old_global=cj,q1=ds1,q4=ds4,later_q4=later,physical_pass=ph['physical_pass'][common])
        write(dest/'COMPLETE.json',dict(binding=binding,outputs={p.name:sha(p) for p in dest.iterdir() if p.name!='COMPLETE.json'}))
        print('ANALYZED',seed,task,'pairs',len(ii),'physical',int(ph['physical_pass'].sum()),flush=True)
    # Blind ordering prevents reviewer seeing label/gate group.
    selected=sorted(selected,key=lambda r:hashlib.sha256(('blind|'+r['new_file']+'|'+r['old_file']).encode()).hexdigest())
    for k,r in enumerate(selected):r['pair_id']='R%03d'%(k+1)
    write(a.out/'AUDIT_SELECTION_PRIVATE.json',dict(pairs=selected),True)
    table(a.out/'R1_COVERAGE.csv',r1);table(a.out/'R1_BY_OLD_TASK.csv',byold);table(a.out/'R2_PHYSICAL_AUDIT.csv',r2);table(a.out/'R3_NONOVERLAP.csv',r3);table(a.out/'R3_TRAJECTORY.csv',traj)
    write(a.out/'R1_R3_COMPLETE.json',dict(status='PASS',seconds=time.time()-t0,cells=9,ai_audit_pairs=len(selected),formal_training_resumed=False))

if __name__=='__main__':main()
