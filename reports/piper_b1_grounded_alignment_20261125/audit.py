"""Independently reproduce all aggregate checks from the public scalar outcomes."""
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):
    return json.loads(p.read_bytes())


def dist(values):
    v=np.asarray(values,float)
    assert np.all(np.isfinite(v))
    return dict(n=len(v),minimum=v.min(),median=np.median(v),p90=np.percentile(v,90),maximum=v.max())


def grouped(rows,pairs,key,mode):
    filtered=[r for r in rows if r['mode']==mode]
    assert len(filtered)==2*len(pairs)
    values=[]
    for a,b in pairs:
        group=[r for r in filtered if (r['recipient'],r['donor']) in ((a,b),(b,a))]
        assert {r['recipient'] for r in group}=={a,b} and len(group)==2
        values.append(np.mean([r[key] for r in group]))
    return np.array(values)


def evaluate(outcome,old,fit):
    pairs=outcome['pairs']
    assert pairs==old['geometry']['pairs'] and len({e for p in pairs for e in p})==32
    episodes=set(old['geometry']['test_episodes'])
    assert episodes.isdisjoint(fit['train_episodes']) and len(episodes)==33 and len(fit['train_episodes'])==50
    base=grouped(outcome['directions'],pairs,'signed_mm','base')
    np.testing.assert_allclose(base,np.array(old['approach_pairs_zero']['baseline']['overhead'])[:,0],atol=1e-7)
    boot=np.random.default_rng(1125).integers(len(pairs),size=(20000,len(pairs)))
    summary={}
    for mode in sorted({r['mode'] for r in outcome['natural']}):
        dirs=[r for r in outcome['directions'] if r['mode']==mode]
        nat=[r for r in outcome['natural'] if r['mode']==mode]
        assert {r['episode'] for r in nat}==episodes and len(nat)==33
        proj=grouped(dirs,pairs,'signed_mm',mode)
        independent=grouped(dirs,pairs,'contact_signed_mm',mode)
        gain=sum(r['toward_new_mm']>0 for r in dirs)
        positive=int(sum(proj>0)); correct=int(sum(proj>1)); wrong=int(sum(proj< -1))
        cp=int(sum(independent>0))
        med=np.median([r['error_mm'] for r in nat])
        base_error=np.median([r['error_mm'] for r in outcome['natural'] if r['mode']=='base'])
        summary[mode]=dict(positive=positive,over1=correct,reverse_over1=wrong,projection=dist(proj),
            signed_by_pair=proj.tolist(),median_difference_ci95_mm=np.percentile(np.median((proj-base)[boot],axis=1),[2.5,97.5]).tolist(),
            independently_positive=cp,independent_median_mm=np.median(independent),toward_new_of32=gain,
            toward_contact_of32=sum(r['toward_contact_mm']>0 for r in dirs),
            natural_closer_image=sum(r['advance_mm']>0 for r in nat),
            natural_closer_contact=sum(r['contact_advance_mm']>0 for r in nat),
            feedback_error_mm=med,natural_contact_gap_mm=np.median([r['contact_end_mm'] for r in nat]),
            natural_contact_advance_mm=np.median([r['contact_advance_mm'] for r in nat]),
            passes=bool(positive>=14 and correct>=12 and wrong<=1 and gain>=24 and cp>=12 and med<=base_error))
    loc={k:dist([r[k] for r in outcome['locations'] if r.get(k) is not None]) for k in
         ('heatmap_px','mid_px','heatmap_contact_mm','label_contact_mm','fused_contact_mm') if k in outcome['locations'][0]}
    return dict(modes=summary,localization=loc)


def extras(data,pairs):
    result={}
    for name,rows in data['field'].items():
        stats={}
        for key in rows[0]:
            if key.endswith('_px') or 'advance_mm' in key:
                scores=[r[key] for r in rows]
                stats[key]=dict(**dist(scores),positive=int(sum(x>0 for x in scores)))
        result[name]=stats
    isolated={}
    for k in (32,52):
        rows=[dict(r,mode='only') for r in data['object_only'] if r['training_cutout']==k]
        s=grouped(rows,pairs,'signed_mm','only')
        isolated[k]=dict(positive=int(sum(s>0)),over1=int(sum(s>1)),median_signed_mm=np.median(s),
                         new_pixel_error=dist([r['new_pixel_error'] for r in rows]))
    return dict(field=result,only_block_changed=isolated)


def audit(root):
    old=read(root.parent/'piper_b1_vision_contrasts_20261124/final.json')
    fit=read(root/'fit.json')
    assert hashlib.sha256((root/'protocol.md').read_bytes()).hexdigest()==fit['protocol_sha256']
    data=read(root/'audit_inputs.json');v2=data['appearance_training'];pairs=old['geometry']['pairs']
    result=dict(first=evaluate(read(root/'outcomes.json'),old,fit),first_checks=extras(data,pairs),
        augmented=evaluate(v2['outcomes'],old,fit),augmented_checks=extras(v2['audit_inputs'],pairs))
    replay=data['raw_replay']
    candidate=replay['mode']
    metrics=result['augmented']['modes'][candidate]
    assert replay['original_checkpoint_sha256']==old['checkpoint_ids']['baseline']
    meta=v2['inspection']
    assert meta['groups']==50 and set(meta['validation_groups']).issubset(fit['train_episodes'])
    assert replay['private_hashes']=={'private_heatmap.pt':meta['heatmap_sha256'],
        'private_policy.npz':fit['policy_sha256'],'private_settle.npz':meta['settle']['sha256']}
    assert (replay['natural_count'],replay['swapped_count'],replay['exact_bypass_count'])==(33,32,65)
    assert replay['max_scalar_difference_mm']<1e-4 and replay['max_gripper_difference']==0
    assert replay['max_height_difference_mm']<1e-4 and replay['scalar_count']==97
    assert replay['missing_feedback_or_wrist_rejected']
    result['candidate']=candidate
    result['offline_accepted']=metrics['passes']
    for label,swapped in [('natural_distances',False),('swapped_distances',True)]:
        rows=[r for r in replay['distances'] if (r['recipient']!=r['donor'])==swapped]
        assert len(rows)==(32 if swapped else 33)
        table={k:dist([r[k] for r in rows]) for k in rows[0] if k.endswith('_mm') and rows[0][k] is not None}
        for target in ('block','contact'):
            gains=[r['initial_to_'+target+'_mm']-r['final_to_'+target+'_mm'] for r in rows]
            table[target+'_advance_mm']=dist(gains)
            table[target+'_closer']=int(np.sum(np.array(gains)>0))
        result[label]=table
    assert result['swapped_distances']['block_closer']==metrics['toward_new_of32']
    assert result['swapped_distances']['contact_closer']==metrics['toward_contact_of32']
    novel=data['independent_visibility']
    assert len(novel)==7 and {v['episode'] for v in novel}.isdisjoint(set(fit['train_episodes'])|set(old['geometry']['test_episodes']))
    visible=[r for r in novel if r['reference_uv'] is not None]
    for r in visible:
        assert abs(np.linalg.norm(np.subtract(r['estimate_uv'],r['reference_uv']))-r['error_px'])<1e-7
    assert len(visible)==2
    result['additional_visibility']=dict(total=len(novel),scorable=len(visible),errors_px=[r['error_px'] for r in visible])
    if 'phase_study' in data:
        phase={}
        for offset in sorted({r['offset'] for r in data['phase_study']}):
            rows=[r for r in data['phase_study'] if r['offset']==offset]
            phase[str(offset)]={k:dist([r[k] for r in rows]) for k in
                ('original_gap_mm','predicted_gap_mm','goal_contact_mm','hold_error_mm','prediction_error_mm')}
            phase[str(offset)]['closer']=sum(r['predicted_gap_mm']<r['original_gap_mm'] for r in rows)
            phase[str(offset)]['stayed_within_1mm_of_initial_gap']=sum(r['predicted_gap_mm']<=r['original_gap_mm']+1 for r in rows)
        result['precontact_phase']=phase
    (root/'summary.json').write_bytes((json.dumps(result,indent=1,allow_nan=False)+'\n').encode())


if __name__=='__main__':
    audit(Path(__file__).resolve().parent)
