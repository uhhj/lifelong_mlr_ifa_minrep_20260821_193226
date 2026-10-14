"""Full-population blocked P1/P2/P3 statistics; never trains a model."""
import argparse
import json
from pathlib import Path
import time
import resource
import numpy as np
from scipy.spatial.distance import cdist
from .common import PROTOCOL,sha,write,table,npz,normalize

def action_distance(new,old,ii,jj,q,scale):
    result=np.empty(len(ii),np.float64)
    for s in range(0,len(ii),8192):
        e=min(s+8192,len(ii));delta=(new[ii[s:e],:q].astype(float)-old[jj[s:e],:q].astype(float))/scale
        result[s:e]=np.sqrt(np.mean(delta*delta,axis=(1,2)))
    return result

def paired_ci(delta,seed=42,repeats=2000):
    delta=np.asarray(delta,float)
    if not len(delta): return dict(n=0,mean=None,ci95=None)
    rng=np.random.RandomState(seed);values=np.mean(delta[rng.randint(len(delta),size=(repeats,len(delta)))],axis=1)
    return dict(n=len(delta),mean=float(delta.mean()),ci95=np.quantile(values,[.025,.975]).tolist(),
                unit='new-task trajectory paired mean; old retrieval pool treated fixed')

def load_all(out,manifest):
    all_values={};task=[];demo=[];time_index=[]
    for rec in manifest['demos']:
        comp=json.loads((out/(rec['id']+'_FEATURE_COMPLETE.json')).read_text())
        assert sha(out/(rec['id']+'_features.npz'))==comp['sha256']
        assert sha(out/(rec['id']+'_raw.npz'))==rec['raw_sha256']
        for suffix in ('raw','features'):
            with np.load(out/(rec['id']+'_'+suffix+'.npz'),allow_pickle=False) as z:
                for key in z.files: all_values.setdefault(key,[]).append(z[key])
        task.extend([rec['task']]*rec['n']);demo.extend([rec['demo']]*rec['n']);time_index.extend(range(rec['n']))
    return {**{k:np.concatenate(v) for k,v in all_values.items()},'task':np.array(task),'demo':np.array(demo),'t':np.array(time_index)}

def retrieval_rows(data,ni,oi,indices,method,stage,beta):
    rows=[]
    for local,j in enumerate(indices):
        i=int(ni[local]);j=int(oi[j]) if j>=0 else -1
        gt=bool(j>=0 and data['target_valid'][i] and data['target_valid'][j])
        rows.append(dict(task=stage,beta=beta,method=method,new_index=i,new_demo=int(data['demo'][i]),new_t=int(data['t'][i]),
            old_index=j,old_task=int(data['task'][j]) if j>=0 else None,old_demo=int(data['demo'][j]) if j>=0 else None,
            old_t=int(data['t'][j]) if j>=0 else None,available=j>=0,
            eef_error_m=float(np.linalg.norm(data['eef'][i]-data['eef'][j])) if j>=0 else None,
            object_relative_vector_error_m=float(np.linalg.norm(data['target_relative'][i]-data['target_relative'][j])) if gt else None,
            object_eef_distance_error_m=float(abs(np.linalg.norm(data['target_relative'][i])-np.linalg.norm(data['target_relative'][j]))) if gt else None,
            object_gt_available=gt))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
    start=time.perf_counter();manifest=json.loads((out/'DATA_MANIFEST.json').read_text());norm=json.loads((out/'NORMALIZATION.json').read_text())
    feature_complete=json.loads((out/'FEATURES_COMPLETE.json').read_text());assert feature_complete['status']=='PASS'
    binding=dict(protocol=sha(out/'PROTOCOL.json'),data_manifest=sha(out/'DATA_MANIFEST.json'),features=sha(out/'FEATURES_COMPLETE.json'),
                 analyzer=sha(__file__),common=sha(Path(__file__).with_name('common.py')))
    write(out/'ANALYSIS_BINDING.json',binding,immutable=True)
    # Output presence alone is insufficient: changed input arrays invalidate resume.
    for rec in manifest['demos']:
        assert sha(out/(rec['id']+'_raw.npz'))==rec['raw_sha256'], 'RAW_ARRAY_CHANGED'
        record=json.loads((out/(rec['id']+'_FEATURE_COMPLETE.json')).read_text())
        assert sha(out/(rec['id']+'_features.npz'))==record['sha256'], 'FEATURE_ARRAY_CHANGED'
    if (out/'ANALYSIS_COMPLETE.json').exists():
        finished=json.loads((out/'ANALYSIS_COMPLETE.json').read_text())
        assert finished['binding']==binding
        for name,digest in finished['outputs'].items(): assert sha(out/name)==digest
        for task in [6,7,8,9]:
            stage=out/('task%d'%task);record=json.loads((stage/'COMPLETE.json').read_text())
            assert record['binding']==binding
            for name,digest in record['outputs'].items(): assert sha(stage/name)==digest
        print('ANALYSIS_ALREADY_VERIFIED_COMPLETE',flush=True);return
    d=load_all(out,manifest);state=(d['state']-norm['state_mean'])/norm['state_std'];scale=np.array(norm['action_scale'])
    va=normalize(d['visual'][:,0]);ve=normalize(d['visual'][:,1]);H=normalize(d['full_H'].reshape(len(state),-1))
    p1=[];p2=[];p3=[];contributions=[];stage_files={};thresholds=[]
    for task in [6,7,8,9]:
        stage=out/('task%d'%task);stage.mkdir(exist_ok=True)
        complete=stage/'COMPLETE.json'
        if complete.exists():
            old=json.loads(complete.read_text());assert old['binding']==binding
            for name,digest in old['outputs'].items(): assert sha(stage/name)==digest
            p1.extend(json.loads((stage/'P1.json').read_text()));p2.extend(json.loads((stage/'P2.json').read_text()))
            p3.extend(json.loads((stage/'P3.json').read_text()));contributions.extend(json.loads((stage/'BY_OLD_TASK.json').read_text()))
            thresholds.extend(json.loads((stage/'THRESHOLDS.json').read_text()));continue
        ni=np.where(d['task']==task)[0];oi=np.where((d['task']<task)&(d['demo']<5))[0];shape=(len(ni),len(oi))
        ds=np.memmap(stage/'Ds.float64',mode='w+',dtype='float64',shape=shape)
        ca=np.memmap(stage/'cos_agent.float64',mode='w+',dtype='float64',shape=shape)
        ce=np.memmap(stage/'cos_eih.float64',mode='w+',dtype='float64',shape=shape)
        stage_start=time.perf_counter()
        for s in range(0,len(ni),64):
            ind=ni[s:s+64];ds[s:s+len(ind)]=cdist(state[ind],state[oi]);ca[s:s+len(ind)]=va[ind]@va[oi].T;ce[s:s+len(ind)]=ve[ind]@ve[oi].T
        ds.flush();ca.flush();ce.flush();tau_s=float(np.quantile(ds,.15))
        hrank=np.empty(len(ni),int);hgate=np.empty(len(ni),int)
        for s in range(0,len(ni),64):
            ind=ni[s:s+64];sim=H[ind]@H[oi].T;hrank[s:s+len(ind)]=sim.argmax(1)
            eligible=ds[s:s+len(ind)]<=tau_s;sim[~eligible]=-np.inf
            choice=sim.argmax(1);choice[~eligible.any(1)]=-1;hgate[s:s+len(ind)]=choice
        stage_p1=[];stage_p2=[];stage_p3=[];stage_contrib=[];stage_thresholds=[]
        for beta in [.3,.5,.7]:
            score=beta*np.asarray(ce)+(1-beta)*np.asarray(ca)
            tau_v=float(np.quantile(score,.85));physical=ds<=tau_s;comparable=physical&(score>=tau_v)
            ii,jj=np.where(comparable);future=d['future_valid'][ni[ii],:4].all(1)&d['future_valid'][oi[jj],:4].all(1)
            i4,j4=ii[future],jj[future];dist4=action_distance(d['future'][ni],d['future'][oi],i4,j4,4,scale)
            assert len(dist4),'NO_COMPARABLE_FUTURE_PAIRS'
            tp,tm=[float(x) for x in np.quantile(dist4,[.25,.75])]
            assert tp<tm,'DEGENERATE_ACTION_QUARTILES'
            counts,edges=np.histogram(dist4,bins=60,range=(0,float(dist4.max())))
            npz(stage/('P1_conditional_action_hist_beta%.1f.npz'%beta),counts=counts,edges=edges,
                quantiles=np.quantile(dist4,[0,.01,.05,.25,.5,.75,.95,.99,1]),
                quantile_levels=np.array([0,.01,.05,.25,.5,.75,.95,.99,1]))
            shared=dist4<=tp;bif=dist4>=tm
            cover=lambda inds:float(len(np.unique(inds))/len(ni))
            scov,pcov,bcov=cover(ii),cover(i4[shared]),cover(i4[bif])
            pregrip=np.any((d['future'][ni[i4],:4,6]>0)!=(d['future'][oi[j4],:4,6]>0),1)
            row=dict(task=task,beta=beta,n_current=len(ni),n_old=len(oi),n_all_pairs=int(np.prod(shape)),tau_s=tau_s,tau_v=tau_v,
                tau_plus=tp,tau_minus=tm,physical_pairs=int(physical.sum()),visual_pairs=int((score>=tau_v).sum()),
                comparable_pairs=len(ii),comparable_q4_pairs=len(i4),share_pairs=int(shared.sum()),bif_pairs=int(bif.sum()),
                state_anchor_coverage=scov,share_anchor_coverage=pcov,bif_anchor_coverage=bcov,
                share_pair_fraction_conditional=float(shared.mean()),bif_pair_fraction_conditional=float(bif.mean()),
                share_pairs_with_gripper_conflict=int((shared&pregrip).sum()),bif_pairs_with_gripper_conflict=int((bif&pregrip).sum()),
                count_gate_primary=bool(scov>=.30 and pcov>=.10 and bcov>=.10),
                count_gate_lower_band=bool(scov>=.30 and pcov>=.08 and bcov>=.08),
                physical_semantic_validity='NOT_ESTABLISHED_BY_QUANTILES',q4_tail_current=int((~d['future_valid'][ni,:4].all(1)).sum()),
                q4_tail_old=int((~d['future_valid'][oi,:4].all(1)).sum()))
            for label,indices in [('state',ii),('share',i4[shared]),('bif',i4[bif])]:
                covered=np.zeros(len(ni),bool);covered[indices]=True
                values=[float(covered[d['demo'][ni]==demo].mean()) for demo in range(10)]
                ci=paired_ci(values)
                row[label+'_trajectory_mean_coverage']=ci['mean']
                row[label+'_trajectory_ci_low']=ci['ci95'][0];row[label+'_trajectory_ci_high']=ci['ci95'][1]
            stage_p1.append(row);stage_thresholds.append(dict(task=task,beta=beta,tau_s=tau_s,tau_v=tau_v,tau_plus=tp,tau_minus=tm))
            for old_task in range(task):
                mask=d['task'][oi[j4]]==old_task
                stage_contrib.append(dict(task=task,beta=beta,old_task=old_task,comparable_q4_pairs=int(mask.sum()),
                    share_pairs=int((mask&shared).sum()),bif_pairs=int((mask&bif).sum()),
                    unique_demo_pairs=len(set(zip(d['demo'][ni[i4[mask]]].tolist(),d['demo'][oi[j4[mask]]].tolist())))))
            # Plot sample is fixed-seed and never used for thresholds or decisions.
            rng=np.random.RandomState(42+task);ps_i,ps_j=np.where(physical)
            ok=d['future_valid'][ni[ps_i],:4].all(1)&d['future_valid'][oi[ps_j],:4].all(1);ps_i,ps_j=ps_i[ok],ps_j[ok]
            pick=rng.choice(len(ps_i),min(2000,len(ps_i)),replace=False);si,sj=ps_i[pick],ps_j[pick]
            npz(stage/('P1_plot_beta%.1f.npz'%beta),visual=score[si,sj],state_distance=np.asarray(ds[si,sj]),
                action=action_distance(d['future'][ni],d['future'][oi],si,sj,4,scale),new_index=ni[si],old_index=oi[sj])
            # P2: both end-to-end and controls that remove state-input advantages.
            visionrank=score.argmax(1);gated=score.copy();gated[~physical]=-np.inf
            cr=gated.argmax(1);cr[~physical.any(1)]=-1
            retrieval=[]
            for method,ranks in [('H_full_cosine',hrank),('C_physical_gate_then_visual',cr),('unmod_visual_only',visionrank),('H_same_physical_gate',hgate)]:
                retrieval.extend(retrieval_rows(d,ni,oi,ranks,method,task,beta))
            table(stage/('P2_retrieval_beta%.1f.csv'%beta),retrieval)
            per_demo=[]
            for method in PROTOCOL['p2_methods']:
                for demo in range(10):
                    selected=[r for r in retrieval if r['method']==method and r['new_demo']==demo]
                    for metric in ['eef_error_m','object_relative_vector_error_m','object_eef_distance_error_m']:
                        vals=[r[metric] for r in selected if r[metric] is not None]
                        per_demo.append(dict(task=task,beta=beta,method=method,demo=demo,metric=metric,n=len(vals),mean=float(np.mean(vals)) if vals else None,
                            median=float(np.median(vals)) if vals else None,p90=float(np.quantile(vals,.9)) if vals else None))
            table(stage/('P2_trajectory_beta%.1f.csv'%beta),per_demo)
            for method,baseline in [('C_physical_gate_then_visual','H_full_cosine'),('unmod_visual_only','H_full_cosine'),('C_physical_gate_then_visual','H_same_physical_gate')]:
                for metric in ['eef_error_m','object_relative_vector_error_m','object_eef_distance_error_m']:
                    # Paired only on common available anchors, then mean within trajectory.
                    left={r['new_index']:r for r in retrieval if r['method']==method};right={r['new_index']:r for r in retrieval if r['method']==baseline}
                    deltas=[]
                    for demo in range(10):
                        values=[left[k][metric]-right[k][metric] for k in left if left[k]['new_demo']==demo and left[k][metric] is not None and right[k][metric] is not None]
                        if values: deltas.append(float(np.mean(values)))
                    stage_p2.append(dict(task=task,beta=beta,method=method,baseline=baseline,metric=metric,**paired_ci(deltas)))
            # P3: fixed q4 thresholds. Main comparisons share a q6-complete pair set.
            common=d['future_valid'][ni[ii],:6].all(1)&d['future_valid'][oi[jj],:6].all(1)
            ci,cj=ii[common],jj[common]
            distances={q:action_distance(d['future'][ni],d['future'][oi],ci,cj,q,scale) for q in range(1,7)}
            single=distances[1]<=tp
            for q in range(1,7):
                validq=d['future_valid'][ni[ii],:q].all(1)&d['future_valid'][oi[jj],:q].all(1)
                s_q=distances[q]<=tp
                future_div=np.zeros(len(ci),bool)
                for h in range(q+1,7): future_div|=distances[h]>=tm
                trajectory_rates=[]
                for demo in range(10):
                    selected=(d['demo'][ni[ci]]==demo)&single
                    if selected.any(): trajectory_rates.append(float((distances[q][selected]>=tm).mean()))
                bootstrap=paired_ci(trajectory_rates)
                stage_p3.append(dict(task=task,beta=beta,q=q,total_comparable_pairs=len(ii),q_available_pairs=int(validq.sum()),
                    pair_truncation_fraction=1-float(validq.mean()),common_q6_pairs=len(ci),q1_shared_count=int(single.sum()),
                    q1_shared_now_bif_count=int((single&(distances[q]>=tm)).sum()),
                    q1_shared_now_bif_rate=float((distances[q][single]>=tm).mean()) if single.any() else None,
                    current_q_shared_count=int(s_q.sum()),later_up_to_6_bif_count=int((s_q&future_div).sum()),
                    later_up_to_6_bif_rate=float(future_div[s_q].mean()) if s_q.any() and q<6 else None,
                    q1_shared_bif_trajectory_mean=bootstrap['mean'],trajectory_n=bootstrap['n'],
                    trajectory_ci_low=bootstrap['ci95'][0] if bootstrap['ci95'] else None,
                    trajectory_ci_high=bootstrap['ci95'][1] if bootstrap['ci95'] else None,
                    interpretation='observed action divergence, not physical false-positive ground truth'))
            npz(stage/('P3_common_beta%.1f.npz'%beta),new_index=ni[ci],old_index=oi[cj],**{'Dq%d'%q:v for q,v in distances.items()})
            del score,gated,physical,comparable
        for name,value in [('P1.json',stage_p1),('P2.json',stage_p2),('P3.json',stage_p3),('BY_OLD_TASK.json',stage_contrib),('THRESHOLDS.json',stage_thresholds)]:
            write(stage/name,value,immutable=True)
        outputs={p.name:sha(p) for p in stage.iterdir() if p.suffix in ('.json','.csv','.npz') and p.name!='COMPLETE.json'}
        write(complete,dict(binding=binding,outputs=outputs,seconds=time.perf_counter()-stage_start),immutable=True)
        p1+=stage_p1;p2+=stage_p2;p3+=stage_p3;contributions+=stage_contrib;thresholds+=stage_thresholds
        status=dict(stage='STATISTICS',task_completed=task,tasks_completed=task-5,tasks_total=4,seconds=time.perf_counter()-start,formal_training_resumed=False)
        write(out/'STATUS.json',status);print(json.dumps(status),flush=True)
        del ds,ca,ce
    table(out/'P1_COVERAGE.csv',p1);table(out/'P3_HORIZON_DIAGNOSTICS.csv',p3);table(out/'P1_BY_OLD_TASK.csv',contributions)
    write(out/'P2_PAIRED_TRAJECTORY_REPORT.json',p2,immutable=True);write(out/'FROZEN_STAGE_THRESHOLDS.json',thresholds,immutable=True)
    write(out/'ANALYSIS_COMPLETE.json',dict(status='PASS',binding=binding,seconds=time.perf_counter()-start,
        outputs={name:sha(out/name) for name in ['P1_COVERAGE.csv','P3_HORIZON_DIAGNOSTICS.csv','P1_BY_OLD_TASK.csv',
            'P2_PAIRED_TRAJECTORY_REPORT.json','FROZEN_STAGE_THRESHOLDS.json']},
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,formal_training_resumed=False),immutable=True)

if __name__=='__main__': main()
