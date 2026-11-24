"""Independently aggregate the 16 fixed disjoint pairs, with checks against previous runs."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_bytes())


def summarize(root):
    inputs=load(root/'audit_inputs.json')
    outcomes=load(root/'outcomes.json')
    fit=load(root/'fit.json')
    baseline=load(root.parent/'piper_b1_vision_contrasts_20261124/final.json')
    previous_root=root.parent/'piper_b1_relative_policy_20261125'
    joint=load(previous_root/'summary.json')
    pairs=inputs['pairs']
    assert pairs==baseline['geometry']['pairs']==joint['pairs'] and len(pairs)==16
    assert len({ep for p in pairs for ep in p})==32
    assert inputs['windows_sha256']==load(previous_root/'data_audit.json')['private_archive_sha256']
    assert inputs['checkpoint_sha256']==baseline['checkpoint_ids']['baseline']
    assert inputs['protocol_sha256']==hashlib.sha256((root/'protocol.md').read_bytes()).hexdigest()
    train=set(inputs['training_episodes'])
    held=set(inputs['heldout_episodes'])
    assert len(train)==50 and len(held)==33 and train.isdisjoint(held)
    assert list(inputs['train_per_offset'].values())==[50]*4
    assert inputs['n_train']==200 and inputs['n_held']==33
    ik=outcomes['pose_reconstruction']
    assert ik['xy_max_mm']<1e-5 and ik['height_max_mm']<1e-5 and ik['angle_max_deg']<1e-5
    assert ik['poses']==8*(33*3+32*5)
    verify=fit['image_to_joint_verification']
    assert verify['natural']==33 and verify['swapped']==32 and verify['max_scalar_difference_mm']<1e-5
    values={}
    table={}
    bootstrap=np.random.default_rng(1125).integers(16,size=(20000,16))
    fields=['signed_mm','orthogonal_mm','amplitude_mm','contact_signed_mm','donor_distance_reduction_mm','separation_mm']
    for mode in ('base','xy','fixed','permuted','wood_only','jaw_only'):
        data=[r for r in outcomes['directions'] if r['mode']==mode]
        assert len(data)==32
        grouped=[]
        for a,b in pairs:
            twins=[r for r in data if (r['recipient'],r['donor']) in ((a,b),(b,a))]
            assert len(twins)==2 and {r['recipient'] for r in twins}=={a,b}
            for r in twins:
                np.testing.assert_allclose(np.hypot(r['signed_mm'],r['orthogonal_mm']),r['amplitude_mm'],atol=1e-6)
            grouped.append([np.mean([r[k] for r in twins]) for k in fields])
        v=np.asarray(grouped)
        table[mode]=v.tolist()
        signed,side,length,contact,reduction,sep=v.T
        np.testing.assert_allclose(sep,baseline['geometry']['pair_separations_mm'],atol=1e-7)
        if mode=='base':
            np.testing.assert_allclose(signed,np.array(baseline['approach_pairs_zero']['baseline']['overhead'])[:,0],atol=1e-7)
        diff=signed-np.asarray(table['base'])[:,0]
        values[mode]=dict(positive=int(sum(signed>1e-8)),numerical_ties=int(sum(abs(signed)<=1e-8)),positive_over_1mm=int(sum(signed>1)),
            negative_over_1mm=int(sum(signed< -1)),median_signed_mm=float(np.median(signed)),
            mean_signed_mm=float(np.mean(signed)),range_mm=[signed.min(),signed.max()],
            median_gain_fraction=float(np.median(signed/sep)),median_amplitude_mm=float(np.median(length)),
            median_abs_cross_axis_mm=float(np.median([abs(r['orthogonal_mm']) for r in data])),
            median_reduction_to_donor_mm=float(np.median(reduction)),donor_closer=int(sum(reduction>0)),
            contact_positive=int(sum(contact>0)),median_contact_projection_mm=float(np.median(contact)),
            median_difference_vs_base_mm=float(np.median(diff)),paired_ci95_mm=np.percentile(np.median(diff[bootstrap],axis=1),[2.5,97.5]).tolist())
    assert max(abs(np.array(table['fixed'])[:,0]))<1e-5
    natural={}
    for mode in ('base','xy','fixed','permuted','measured'):
        rows=[r for r in outcomes['natural'] if r['mode']==mode]
        assert len(rows)==33 and {r['episode'] for r in rows}==held
        natural[mode]={k:float(np.median([r[k] for r in rows])) for k in
            ('feedback_endpoint_error_mm','travel_mm','distance_to_block_mm','advance_mm')}
        natural[mode]['closer']=sum(r['advance_mm']>0 for r in rows)
    assert natural['measured']['feedback_endpoint_error_mm']==0
    hold_error=natural['measured']['travel_mm']
    evidence=inputs['natural_expert_pairs']
    assert [r['pair'] for r in evidence]==pairs
    real=np.array([r['measured_displacement_difference_projected_mm'] for r in evidence])
    return dict(pairs=pairs,pair_columns=fields,pair_means=table,paired=values,natural_medians=natural,
        hold_feedback_endpoint_error_mm=hold_error,expert_natural_pair_check=dict(positive=int(sum(real>0)),
            median_projected_difference_mm=float(np.median(real)),over_1mm=int(sum(real>1)),
            median_start_separation_mm=float(np.median([r['original_start_separation_mm'] for r in evidence]))))


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    (root/'summary.json').write_bytes((json.dumps(summarize(root),indent=1,allow_nan=False)+'\n').encode())
