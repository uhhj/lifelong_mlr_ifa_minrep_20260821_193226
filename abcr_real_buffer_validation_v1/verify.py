"""Independent scalar checks of archived pair identity, physics, labels and future timing."""
import argparse,csv,math
from pathlib import Path
import numpy as np
from .common import read,write,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();pilot=a.repo/'reports/abcr_object_pilots_v1'
    cfg=read(a.out/'PROTOCOL.json');th=read(pilot/'FROZEN_STAGE_THRESHOLDS.json');scale=read(pilot/'NORMALIZATION.json')['action_scale'];raw={};checked=0;err=0.;future_checked=0
    hdf5_assets=[r for r in read(pilot/'DATA_MANIFEST.json')['assets'] if r['task']>=6]
    for r in hdf5_assets:assert sha(r['path'])==r['sha256'],('RAW_HDF5_CHANGED',r['path'])
    global_records=[]
    for r in read(pilot/'DATA_MANIFEST.json')['demos']:
        if r['task']>=6:global_records.extend((r['task'],r['demo'],t) for t in range(r['n']))
    summaries=list(csv.DictReader((a.out/'R1_COVERAGE.csv').open()));phy=list(csv.DictReader((a.out/'R2_PHYSICAL_AUDIT.csv').open()))
    for s in read(a.out/'STAGE_MANIFEST.json')['stages']:
        root=a.out/('seed%d_task%d'%(s['seed'],s['task']));complete=read(root/'COMPLETE.json')
        for name,h in complete['outputs'].items():assert sha(root/name)==h
        z=np.load(root/'candidate_pairs.npz');order=read(root/'record_order.json');t=next(r for r in th if r['task']==s['task'] and r['beta']==.5)
        for idx in np.linspace(0,len(z['share'])-1,min(100,len(z['share'])),dtype=int):
            n,o=order['new'][int(z['new_local'][idx])],order['old'][int(z['old_local'][idx])];vals=[]
            for r in [n,o]:
                key=(r['task'],r['demo'])
                if key not in raw:raw[key]=np.load(pilot/('task%d_demo%d_raw.npz'%key))
                vals.append(raw[key])
            na,oa=vals;ni,oi=n['anchor_t'],o['anchor_t'];ss=0.
            for h in range(1,5):
                for c in range(7):ss+=((float(na['actions'][ni+h,c])-float(oa['actions'][oi+h,c]))/scale[c])**2
            distance=math.sqrt(ss/28);err=max(err,abs(distance-float(z['action_distance_q4'][idx])));assert err<1e-12
            assert bool(z['share'][idx])==(distance<=t['tau_plus']);assert bool(z['bif'][idx])==(distance>=t['tau_minus'])
            eef=math.sqrt(sum((float(na['eef'][ni,c])-float(oa['eef'][oi,c]))**2 for c in range(3)));assert abs(eef-z['eef_m'][idx])<1e-12
            checked+=1
        for gate,mask in [('quantile',np.ones(len(z['share']),bool)),('physical',z['physical_pass']),('physical_object_proxy',z['object_proxy_pass'])]:
            row=next(r for r in summaries if int(r['seed'])==s['seed'] and int(r['task'])==s['task'] and r['gate']==gate)
            for label in ['share','bif']:
                m=mask&z[label];assert int(row[label+'_pairs'])==int(m.sum());assert abs(float(row[label+'_coverage'])-len(set(z['new_local'][m]))/s['n_current'])<1e-15
        zz=np.load(root/'nonoverlap.npz');assert len(zz['q1'])==len(zz['q4'])==len(zz['later_q4'])
        for idx in np.linspace(0,len(zz['q1'])-1,min(30,len(zz['q1'])),dtype=int):
            n,o=global_records[zz['new_global'][idx]],global_records[zz['old_global'][idx]]
            for rec in [n,o]:
                if rec[:2] not in raw:raw[rec[:2]]=np.load(pilot/('task%d_demo%d_raw.npz'%rec[:2]))
            for key,hs in [('q1',range(1,2)),('q4',range(1,5)),('later_q4',range(5,9))]:
                values=[((float(raw[n[:2]]['actions'][n[2]+h,c])-float(raw[o[:2]]['actions'][o[2]+h,c]))/scale[c])**2 for h in hs for c in range(7)]
                assert abs(math.sqrt(sum(values)/len(values))-zz[key][idx])<1e-12
            future_checked+=1
    labels=list(csv.DictReader((a.out/'semantic_audit/AI_SEMANTIC_LABELS.csv').open()));assert len(labels)==72 and len({r['pair_id'] for r in labels})==72
    for r in labels:assert sha(a.out/'semantic_audit'/(r['pair_id']+'.png'))==r['image_sha256'] and r['annotator']=='AI assistant; not human'
    r4=list(csv.DictReader((a.out/'R4_RESULTS.csv').open()));r4complete=read(a.out/'R4_COMPLETE.json');assert len(r4)==r4complete['n']
    for r in r4:
        z=read(a.out/('R4_'+r['pair_id']+'.json'));assert all(z['restored'].values());assert int(r['optimizer_delta'])==int(r['scheduler_delta'])==1
    for task in [7,8]:
        m=read(a.out/('R4_TASK%d_MANIFEST.json'%task));assert sha(m['anchor'])==m['anchor_sha']
        p=a.out/('R4_TASK%d_PARITY.json'%task)
        if p.exists():assert all(read(p)['checks'].values())
    for name,h in read(a.out/'R4_RUNTIME_PROVENANCE.json')['files'].items():assert sha(a.repo/name)==h
    write(a.out/'INDEPENDENT_CHECK.json',dict(status='PASS',raw_hdf5_sha_reverified=len(hdf5_assets),scalar_pairs_checked=checked,nonoverlap_future_pairs_checked=future_checked,action_distance_max_abs_error=err,semantic_images_hash_verified=len(labels),r4_rows=len(r4),formal_sources_and_anchor_sha_unchanged=True,scope='independent scalar arithmetic and saved-state evidence; NOT independent human annotation or different environment',formal_training_resumed=False))

if __name__=='__main__':main()
